#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// for debug
#include <iostream>



template <typename scalar_t>
__global__ void markonv_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> Kernel_Full_4DTensor,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output
) {
  // get sizes
  const auto batch_size = input.size(0);
  const auto channel_size = input.size(1);
  const auto input_sequence_length = input.size(2);
  const auto kernel_length = Kernel_Full_4DTensor.size(0);
  const auto kernel_number = Kernel_Full_4DTensor.size(3);
  // temp_A = input_sequence_position
  // temp_B = kernel_index * input_sequence_length
  // temp_C = batch_index * kernel_number * input_sequence_length
  // mixed_index = temp_A + temp_B + temp_C
  // Use the following theorem to deduce the index: 
  //// [1]: temp_A + temp_B < output_sequence_length + temp_B = (kernel_index + 1) * output_sequence_length <= kernel_number * output_sequence_length
  //// [1->2.1]: mixed_index / (kernel_number * output_sequence_length) = batch_index
  //// [1->2.2]: mod(mixed_index, kernel_number * output_sequence_length) = temp_A + temp_B
  //// [3] temp_A < output_sequence_length
  //// [3->4.1] (temp_A + temp_B) / output_sequence_length = kernel_index
  //// [3->4.2] mod(temp_A + temp_B, output_sequence_length) = temp_A
  const auto output_sequence_length = input_sequence_length - kernel_length;
  const int mixed_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (mixed_index < batch_size * kernel_number * output_sequence_length){
  // batch index
  const int batch_index = mixed_index / (kernel_number * output_sequence_length);
  // kernel index
  const int temp_A_and_temp_B = mixed_index - batch_index * kernel_number * output_sequence_length;
  const int kernel_index = temp_A_and_temp_B / output_sequence_length;
  // input sequence position 
  const int input_sequence_position = temp_A_and_temp_B - kernel_index * output_sequence_length;
  // Compute the Pvalue
  if (input_sequence_position < output_sequence_length){
    scalar_t Pvalue = 0;
    scalar_t temp_Pvalue_1 = 1.0;
    scalar_t temp_Pvalue_2 = 1.0;
    scalar_t temp_Pvalue_3 = 1.0;

    for(int kernel_position=0; kernel_position < kernel_length; kernel_position++){
      for (int channel1_index=0; channel1_index < channel_size; channel1_index++){
	for (int channel2_index=0; channel2_index < channel_size; channel2_index++){
	  temp_Pvalue_1 = input[batch_index][channel1_index][input_sequence_position + kernel_position];
	  temp_Pvalue_2 = input[batch_index][channel2_index][(input_sequence_position + 1) + kernel_position];
	  temp_Pvalue_3 = Kernel_Full_4DTensor[kernel_position][channel1_index][channel2_index][kernel_index];
	  Pvalue = Pvalue + temp_Pvalue_1 * temp_Pvalue_2 * temp_Pvalue_3;
	}
      }
    }
    output[batch_index][kernel_index][input_sequence_position] = Pvalue;       
  }
  }
}

std::vector<torch::Tensor> markonv_cuda_forward(
    torch::Tensor input, /* shape: [batch_size, channel, sequence_length] */
    torch::Tensor Kernel_Full_4DTensor, /* shape: [kernel_length, channel, channel, kernel_number] */
    torch::Tensor output /* output shape: [batch_size, kernel_number, sequence_length - kernel_length]*/
) {  
  const auto batch_size = input.size(0);
  const auto kernel_number = Kernel_Full_4DTensor.size(3);
  const auto input_sequence_length = input.size(2);
  const auto kernel_length = Kernel_Full_4DTensor.size(0);
  const auto output_sequence_length = input_sequence_length - kernel_length;

  const int threads = 512;
  const dim3 blocks_dim3((batch_size * kernel_number * output_sequence_length + threads - 1) / threads, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "markonv_forward_cuda", ([&] {
    markonv_cuda_forward_kernel<scalar_t><<<blocks_dim3, threads>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        Kernel_Full_4DTensor.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
    );
  }));

  
  //cudaError_t cuda_last_error_forward = cudaGetLastError();


  return {output};
}



// backward part


template <typename scalar_t>
__global__ void markonv_cuda_backward_grad_input_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> Kernel_Full_4DTensor,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_input
) {
  // get sizes
  const auto batch_size = input.size(0);
  const auto channel_size = input.size(1);
  const auto input_sequence_length = input.size(2);
  const auto kernel_length = Kernel_Full_4DTensor.size(0);
  const auto kernel_number = Kernel_Full_4DTensor.size(3);
  // mixed index 
  const int mixed_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (mixed_index < batch_size * channel_size * input_sequence_length){
  // batch_index
  const int batch_index = mixed_index / (channel_size * input_sequence_length);
  const int temp_A_and_temp_B = mixed_index - batch_index * channel_size * input_sequence_length;
  // channel_index
  const int channel_index = temp_A_and_temp_B / input_sequence_length;
  // input_sequence_position
  const int input_sequence_position = temp_A_and_temp_B - channel_index * input_sequence_length;
  // Compute the input grad
  if (input_sequence_position < input_sequence_length){  
    // clear the grad first
    grad_input[batch_index][channel_index][input_sequence_position] = 0;
    /* 
       First point (valid only if input_sequence_position + 1 <= input_sequence_length - 1):
       channel_1_index = channel_index as first point
       iterate over the following:
       temp_channel_2_index as second point \in 0, ..., channel_size - 1
       temp_kernel_index \in 0, ..., kernel_number - 1
       temp_kernel_position \in max(0, (input_sequence_position + 1) + kernel_length - input_sequence_length), ..., min(kernel_length - 1, input_sequence_position)
       temp_output_sequence_position = input_sequence_position - temp_kernel_position;
       temp_grad_input_first_point += input[batch_index][temp_channel_2_index][input_sequence_position + 1] * Kernel_Full_4DTensor[temp_kernel_position][channel_1_index][temp_channel_2_index][temp_kernel_index] * grad_output[batch_index][temp_kernel_index][temp_output_sequence_position];
       Final update: grad_input[batch_index][channel_1_index][input_sequence_position] += temp_grad_input_first_point;
    */
    scalar_t temp_grad_input_first_point = 0;
    const int channel_1_index = channel_index;
    if (input_sequence_position + 1 <= input_sequence_length - 1){
      for (int temp_channel_2_index=0; temp_channel_2_index < channel_size; temp_channel_2_index++){	      
	for (int temp_kernel_index=0; temp_kernel_index < kernel_number; temp_kernel_index++){
	  for (int temp_kernel_position=max(0, (input_sequence_position + 1) + kernel_length - input_sequence_length); 
	       temp_kernel_position <= min(kernel_length - 1, input_sequence_position); 
	       temp_kernel_position++){
	    int temp_output_sequence_position = input_sequence_position - temp_kernel_position;
	    temp_grad_input_first_point += input[batch_index][temp_channel_2_index][input_sequence_position + 1] * Kernel_Full_4DTensor[temp_kernel_position][channel_1_index][temp_channel_2_index][temp_kernel_index] * grad_output[batch_index][temp_kernel_index][temp_output_sequence_position];
	  }
	}
      }
      grad_input[batch_index][channel_1_index][input_sequence_position] += temp_grad_input_first_point;
    }



    /* 
       Second point (valid only if input_sequence_position >= 1):
       channel_2_index = channel_index as first point
       iterate over the following:
       temp_channel_1_index as first point \in 0, ..., channel_size - 1
       temp_kernel_index \in 0, ..., kernel_number - 1
       temp_kernel_position \in max(0, (input_sequence_position - 1 + 1) + kernel_length - input_sequence_length), ..., min(kernel_length - 1, input_sequence_position - 1)
       temp_output_sequence_position = ( input_sequence_position - 1 ) - temp_kernel_position;
       temp_grad_input_second_point += input[batch_index][temp_channel_1_index][input_sequence_position - 1] * Kernel_Full_4DTensor[temp_kernel_position][temp_channel_1_index][channel_2_index][temp_kernel_index] * grad_output[batch_index][temp_kernel_index][temp_output_sequence_position];
       Final update: grad_input[batch_index][channel_2_index][input_sequence_position] += temp_grad_input_second_point;
    */
    scalar_t temp_grad_input_second_point = 0;
    const int channel_2_index = channel_index;
    if (input_sequence_position >= 1){
      for (int temp_channel_1_index=0; temp_channel_1_index < channel_size; temp_channel_1_index++){	      
	for (int temp_kernel_index=0; temp_kernel_index < kernel_number; temp_kernel_index++){
	  for (int temp_kernel_position=max(0, (input_sequence_position - 1 + 1) + kernel_length - input_sequence_length);
	       temp_kernel_position <= min(kernel_length - 1, input_sequence_position - 1);
	       temp_kernel_position++){
	    int temp_output_sequence_position = (input_sequence_position - 1) - temp_kernel_position;
	    temp_grad_input_second_point += input[batch_index][temp_channel_1_index][input_sequence_position - 1] * Kernel_Full_4DTensor[temp_kernel_position][temp_channel_1_index][channel_2_index][temp_kernel_index] * grad_output[batch_index][temp_kernel_index][temp_output_sequence_position];
	  }
	}
      }
      grad_input[batch_index][channel_2_index][input_sequence_position] += temp_grad_input_second_point;
    }

    // Finish
  }
  }
}



template <typename scalar_t>
__global__ void markonv_cuda_backward_grad_Kernel_Full_4DTensor_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> Kernel_Full_4DTensor,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_Kernel_Full_4DTensor
) {
  // get sizes
  const auto batch_size = input.size(0);
  const auto channel_size = input.size(1);
  const auto input_sequence_length = input.size(2);
  const auto kernel_length = Kernel_Full_4DTensor.size(0);
  const auto kernel_number = Kernel_Full_4DTensor.size(3);
  const auto output_sequence_length = input_sequence_length - kernel_length ;
  // mixed_index
  const int mixed_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (mixed_index < kernel_number * channel_size * channel_size * kernel_length){
  // kernel_index
  const int kernel_index = mixed_index / (channel_size * channel_size * kernel_length);
  // channel_2_index
  const int temp_A_and_temp_B_and_temp_C = mixed_index - kernel_index * channel_size * channel_size * kernel_length;
  const int channel_2_index = temp_A_and_temp_B_and_temp_C / (channel_size * kernel_length);
  // channel_1_index
  const int temp_A_and_temp_B = temp_A_and_temp_B_and_temp_C - channel_2_index *channel_size* kernel_length;
  const int channel_1_index = temp_A_and_temp_B / kernel_length;
  // kernel_position
  const int kernel_position = temp_A_and_temp_B - channel_1_index * kernel_length;

    // Compute the kernel grad
    scalar_t temp_kernel_grad = 0;
    for (int temp_batch_index=0; temp_batch_index < batch_size; temp_batch_index++){
      for (int temp_output_sequence_position=0; temp_output_sequence_position < output_sequence_length; temp_output_sequence_position++){
        int temp_input_sequence_position = temp_output_sequence_position;
        temp_kernel_grad += input[temp_batch_index][channel_1_index][temp_input_sequence_position + kernel_position] *
    input[temp_batch_index][channel_2_index][(temp_input_sequence_position + 1) + kernel_position] *
    grad_output[temp_batch_index][kernel_index][temp_output_sequence_position];
      }
    }
    grad_Kernel_Full_4DTensor[kernel_position][channel_1_index][channel_2_index][kernel_index] = temp_kernel_grad;
  }
}

std::vector<torch::Tensor> markonv_cuda_backward(
    torch::Tensor grad_output,     /* shape: [batch_size, kernel_number, sequence_length - kernel_length]*/
    torch::Tensor input, /* shape: [batch_size, channel, sequence_length] */
    torch::Tensor Kernel_Full_4DTensor /* shape: [kernel_length, channel, channel, kernel_number] */) {
  
  const auto batch_size = input.size(0);
  const auto channel_size = input.size(1);
  const auto input_sequence_length = input.size(2);
  const auto kernel_length = Kernel_Full_4DTensor.size(0);
  const auto kernel_number = Kernel_Full_4DTensor.size(3);

  auto grad_input = torch::zeros_like(input);
  auto grad_Kernel_Full_4DTensor = torch::zeros_like(Kernel_Full_4DTensor);

  const int threads = 512; 


  const int threads_grad_input = threads;
  const dim3 blocks_grad_input_dim3((batch_size * channel_size * input_sequence_length + threads - 1) / threads, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "markonv_backward_cuda_grad_input", ([&] {
    markonv_cuda_backward_grad_input_kernel<scalar_t><<<blocks_grad_input_dim3, threads_grad_input>>>(
        grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        Kernel_Full_4DTensor.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
    );
  }));

  
  //cudaError_t cuda_last_error_grad_input = cudaGetLastError();

  const int threads_grad_kernel = threads;
  const dim3 blocks_grad_kernel_dim3((kernel_length * channel_size * channel_size * kernel_number + threads - 1) / threads, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "markonv_backward_cuda_grad_Kernel_Full_4DTensor", ([&] {
    markonv_cuda_backward_grad_Kernel_Full_4DTensor_kernel<scalar_t><<<blocks_grad_kernel_dim3, threads_grad_kernel>>>(
        grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        Kernel_Full_4DTensor.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_Kernel_Full_4DTensor.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
    );
  }));

  
  //cudaError_t cuda_last_error_grad_kernel = cudaGetLastError();



  return {grad_input, grad_Kernel_Full_4DTensor};
}


/*int main () {

  int batch_size = 3;
  int channel_size = 4;
  int input_sequence_length = 15;
  int kernel_length = 5;
  int kernel_number = 8;

  torch::Tensor input = torch::arange((batch_size * channel_size * input_sequence_length) + 1.0).reshape({batch_size, channel_size, input_sequence_length});
  torch::Tensor Kernel_Full_4DTensor = torch::arange(kernel_length * channel_size * channel_size * kernel_number).reshape({kernel_length, channel_size, channel_size, kernel_number});
  //auto output = markonv_cuda_forward(input, Kernel_Full_4DTensor);
  return 0;
}
*/
