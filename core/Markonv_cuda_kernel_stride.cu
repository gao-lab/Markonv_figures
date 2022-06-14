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
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output,
    int stride
) {

  // In the deduction below, all coordinates are zero-based, DIVISION(a, b) means dividing a by b to get the exact real number (no rounding, flooring, or ceiling), and int(a/b) means C-style division (rounding to the largest integer that does not exceed DIVISION(a, b); in other words, int(a/b) = MATH.FLOOR(DIVISION(a, b)) )
  
  // The complete proof of how we construct:
  /*
    1. input_sequence_position = output_sequence_position * stride
    2. output_sequence_length = MATH.FLOOR( DIVISION( (input_sequence_length - kernel_length), stride ) ) + 1
    3. for a given thread index: mixed_index = blockIdx.x * blockDim.x + threadIdx.x, we have
    3.1. boundary condition: mixed_index <= batch_size * kernel_number * output_sequence_length - 1
    3.2. batch_index = int(mixed_index/(kernel_number * output_sequence_length))
    3.3. kernel_index = int( (temp_A + temp_B)/output_sequence_length ) = int( (mixed_index - batch_index * kernel_number * output_sequence_length)/output_sequence_length )
    3.4. output_sequence_position = mixed_index - batch_index * kernel_number * output_sequence_length - kernel_index * output_sequence_length
   */
  

  
  // Definition of the mapping `input_sequence_position -> output_sequence_position` when stride is considered
  /*


    output_sequence_position is defined on all input_sequence_position values that satisfy the following two:
    1. MATH.MOD(input_sequence_position, stride) == 0 (or in k-mer form: input_sequence_position = stride * k , where k is a nonnegative integer )
    2. input_sequence_END_OF_CONV_position <= input_sequence_position_LARGEST, where 
       (1) input_sequence_END_OF_CONV_position is the end (rightmost) input-sequence-position of the convolution whose start position is input_sequence_position (i.e., input_sequence_END_OF_CONV_position := input_sequence_position + kernel_length - 1)
       (2) input_sequence_position_LARGEST is the largest input_sequence_position (i.e., input_sequence_position_LARGEST:= input_sequence_length - 1 - 1)
    This set of criteria could be rewritten as:
    - stride * k + kernel_length - 1 <= input_sequence_length - 1 - 1 , where k is a nonnegative integer
    Which can be further simplified into:
    - stride * k <= (input_sequence_length 1 - kernel_length) , where k is a nonnegative integer
       

    output_sequence_position := DIVISION(input_sequence_position, stride)
   */
  
  
  // The inverse of the mapping `input_sequence_position -> output_sequence_position` when stride is considered
  /*

    input_sequence_position is defined on all output_sequence_position values

    input_sequence_position = output_sequence_position * stride
   */

  // The range of output_sequence_position (i.e, 0 and output_sequence_length) when stride is considered
  /*
    output_sequence_length := output_sequence_position_LARGEST + 1 , where output_sequence_position_LARGEST is the largest output_sequence_position

    (<=>)
    {by definition of output_sequence_position (how it is computed), the nature of monotonically increasing of DIVISION on numerator (why the largest 'out' corresponds to the largest 'in'), and the range of preimage of output_sequence_position (how the 'in' is restricted)}
    output_sequence_position_LARGEST := DIVISION(input_sequence_position_LARGEST_AND_VALID, stride), where input_sequence_position_LARGEST_AND_VALID is the largest input_sequence_position whose output_sequence_position is defined

    (<=>)
    {by definition of preimage of the `input_sequence_position -> output_sequence_position` mapping, and the nature of monotonically increasing of stride*k computation on k}
    input_sequence_position_LARGEST_AND_VALID :=  stride * k_LARGEST, where k_LARGEST is the largest k that satisfies
    - stride * k <= (input_sequence_length - 1 - kernel_length) , where k is a nonnegative integer

    (<=>)
    Therefore, k_LARGEST satisfies the following two:
    1. stride * k_LARGEST <= input_sequence_length - kernel_length
    2. stride * (k_LARGEST + 1) > input_sequence_length - kernel_length

    (<=>)
    Therefore, k_LARGEST satisfies the following two:
    1. (input_sequence_length - kernel_length) - (stride * k_LARGEST) >= 0
    2. (input_sequence_length - kernel_length) - (stride * k_LARGEST) < stride

    (<=>)
    Because stride > 0, we have
    1. (input_sequence_length - kernel_length)/stride - k_LARGEST >= 0
    2. (input_sequence_length - kernel_length)/stride - k_LARGEST < 1
    Or written in one-line, focusing on k_LARGEST:
    (input_sequence_length - kernel_length)/stride - 1 < k_LARGEST <= (input_sequence_length - kernel_length)/stride

    (<=>)
    {by part of definition of k_LARGEST (it is a nonnegative integer) and definition of MATH.FLOOR}
    k_LARGEST = MATH.FLOOR( DIVISION( (input_sequence_length - kernel_length), stride ) )
    
    Stitching all above gives:
      output_sequence_length 
    = output_sequence_position_LARGEST + 1 
    = DIVISION(input_sequence_position_LARGEST_AND_VALID, stride) + 1
    = DIVISION(stride * k_LARGEST, stride) + 1
    = DIVISION(stride * MATH.FLOOR( DIVISION( (input_sequence_length - kernel_length - 1), stride ) ) - 1 , stride) + 1
    = MATH.FLOOR( DIVISION( (input_sequence_length - kernel_length - 1), stride ) ) + 1
    
    When stride == 1,
    output_sequence_length = input_sequence_length - kernel_length = MATH.CEIL(input_sequence_length - kernel_length)
    else,
    output_sequence_length = input_sequence_length - kernel_length = MATH.FLOOR( DIVISION( (input_sequence_length - kernel_length), stride ) ) - MATH.FLOOR( DIVISION( 1, stride ) ) + 1 = MATH.CEIL(input_sequence_length - kernel_length)

    i.e., output_sequence_length = MATH.FLOOR( DIVISION( (input_sequence_length - kernel_length - 1), stride ) ) + 1
   */
  
  // Indexing of threads:
  /*
    temp_A := output_sequence_position 
    temp_B := kernel_index * output_sequence_length
    temp_C := batch_index * kernel_number * output_sequence_length
    mixed_index := temp_A + temp_B + temp_C

    Immediate results:
    mixed_index = output_sequence_position + kernel_index * output_sequence_length + batch_index * kernel_number * output_sequence_length
    temp_A + temp_B = mixed_index - batch_index * kernel_number * output_sequence_length
    temp_A = (temp_A + temp_B) - kernel_index * output_sequence_length
   */

  // Range of indices:
  /*
    output_sequence_position >= 0 
    output_sequence_position <= output_sequence_length - 1 < output_sequence_length
    kernel_index >= 0
    kernel_index <= kernel_number - 1 < kernel_number
    batch_index >= 0
    batch_index <= batch_size - 1 < batch_size
   */
  
  // Construction of mixed_index from multi-dimensional tensor indices
  /*
    [0]:    output_sequence_length 
         =  MATH.CEIL( (input_sequence_length - kernel_length)/stride )  
    [1]:    temp_A + temp_B 
         {by definition of temp_A}
         =  output_sequence_position + temp_B
	 {by range of output_sequence_position}
	 <= (output_sequence_length - 1) + temp_B 
	 {by definition of temp_B}
	 =  (output_sequence_length - 1) + kernel_index * output_sequence_length
	 {by axiom}
	 =  (kernel_index + 1) * output_sequence_length - 1
	 {by range of kernel_index}
	 <= kernel_number * output_sequence_length - 1

	 i.e., temp_A + temp_B <= kernel_number * output_sequence_length - 1 < kernel_number * output_sequence_length
    [2]:    mixed_index = temp_A + temp_B + temp_C
         {by replacing `temp_A + temp_B` with conclusion of [1]}
	 <= kernel_number * output_sequence_length - 1 + temp_C
	 {by definition of temp_C}
	 =  kernel_number * output_sequence_length - 1 + batch_index * kernel_number * output_sequence_length
	 {by axiom}
	 =  (batch_index + 1) * kernel_number * output_sequence_length - 1
	 {by range of batch_index}
	 <= batch_size * kernel_number * output_sequence_length - 1

	 i.e., mixed_index = temp_A + temp_B + temp_C <= batch_size * kernel_number * output_sequence_length - 1
  */

  // Use the following steps to deduce the multi-dimensional tensor indices
  /*
     Here (again), DIVISION(a, b) means dividing a by b to get the exact real number (no rounding), while int(a/b) means C-style division (rounding to the largest integer that does not exceed DIVISION(a, b); in other words, int(a/b) = MATH.FLOOR(DIVISION(a, b)) )
     We also use the fact of flooring: for a real number x with 0<x<1, MATH.FLOOR(x) = 0
     [3]:   DIVISION(mixed_index, (kernel_number * output_sequence_length) )
         {by definition of mixed_index, temp_A, and temp_B}
	 =  DIVISION( (temp_A + temp_B + batch_index * kernel_number * output_sequence_length), (kernel_number * output_sequence_length) )
	 {by axiom}
	 =  DIVISION( (temp_A + temp_B), (kernel_number * output_sequence_length) ) + 
            DIVISION( (batch_index * kernel_number * output_sequence_length), (kernel_number * output_sequence_length) )
	 {by conclusion of [2] on TERM 1, range of kernel_index on TERM 2, and axiom on TERM 3}
	 =  some_real_number_x1_between_0_and_1 +
	    batch_index

	 Therefore, int( mixed_index/(kernel_number * output_sequence_length) )
	 {by definition of `int(a/b)`}
         = MATH.FLOOR(DIVISION( mixed_index, (kernel_number * output_sequence_length) )) 
	 {by the fact of flooring}
	 = batch_index

     [4]:   temp_A + temp_B
         {by immediate results}
         =  mixed_index - batch_index * kernel_number * output_sequence_length
	 Here, we explcitly compute `temp_A + temp_B`, because it will be used twice below (in [5] and [6])

     [5]:   DIVISION( (temp_A + temp_B), output_sequence_length )
         {by definition of `temp_A + temp_B`}
         =  DIVISION( (output_sequence_position + kernel_index * output_sequence_length) , output_sequence_length)
	 {by axiom}
	 =  DIVISION( output_sequence_position, output_sequence_length) +
	    DIVISION( kernel_index * output_sequence_length, output_sequence_length)
	 {by conclusion of [1] on TERM 1, and axiom on TERM 2}
	 =  some_real_number_x2_between_0_and_1 +
	    kernel_index

	 Therefore, int( (temp_A + temp_B)/output_sequence_length )
	 {by definition of `int(a/b)`}
         = MATH.FLOOR(DIVISION( (temp_A + temp_B), output_sequence_length )) 
	 {by the fact of flooring}
	 = kernel_index

     [6]:   output_sequence_position
         {by definition of temp_A}
         =  temp_A
	 {by immediate results}
	 =  (temp_A + temp_B) - kernel_index * output_sequence_length
	 {by [4]}
	 =  mixed_index - batch_index * kernel_number * output_sequence_length - kernel_index * output_sequence_length
   */

  
   // This also proves that the mapping [mixed_index] -> [batch_index, kernel_index, output_sequence_position, input_sequence_position] is bijective, i.e.
   /*
     1. It is one-to-one / injective, i.e., two different [mixed_index]'s will not map to the same [batch_index, kernel_index, output_sequence_position, input_sequence_position]
     2. It is surjective, i.e., for each [batch_index, kernel_index, output_sequence_position, input_sequence_position], there is always at least one [mixed_index] that maps to it
    */
  
  const int mixed_index = blockIdx.x * blockDim.x + threadIdx.x;

  // get sizes
  const auto batch_size = input.size(0);
  const auto channel_size = input.size(1);
  const auto input_sequence_length = input.size(2);
  const auto kernel_length = Kernel_Full_4DTensor.size(0);
  const auto kernel_number = Kernel_Full_4DTensor.size(3);
  const auto output_sequence_length = output.size(2);

  
  if (mixed_index < batch_size * kernel_number * output_sequence_length){
    // batch index
    const int batch_index = mixed_index / (kernel_number * output_sequence_length);
    // kernel index
    const int temp_A_and_temp_B = mixed_index - batch_index * kernel_number * output_sequence_length;
    const int kernel_index = temp_A_and_temp_B / output_sequence_length;
    // input sequence position
    const int output_sequence_position = temp_A_and_temp_B - kernel_index * output_sequence_length;
    const int input_sequence_position = output_sequence_position * stride;
    // Compute the Pvalue

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
      output[batch_index][kernel_index][output_sequence_position] = Pvalue;       

  }
}

std::vector<torch::Tensor> markonv_cuda_forward(
    torch::Tensor input, /* shape: [batch_size, channel, sequence_length] */
    torch::Tensor Kernel_Full_4DTensor, /* shape: [kernel_length, channel, channel, kernel_number] */
    torch::Tensor output /* output shape: [batch_size, kernel_number, MATH.FLOOR( DIVISION( (input_sequence_length - kernel_length - 1), stride ) ) + 1]*/,
    int stride
) {  
  const auto batch_size = input.size(0);
  const auto kernel_number = Kernel_Full_4DTensor.size(3);
  const auto input_sequence_length = input.size(2);
  const auto kernel_length = Kernel_Full_4DTensor.size(0);
  const auto output_sequence_length = output.size(2);

  const int threads = 512;
  const dim3 blocks_dim3((batch_size * kernel_number * output_sequence_length + threads - 1) / threads, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "markonv_forward_cuda", ([&] {
    markonv_cuda_forward_kernel<scalar_t><<<blocks_dim3, threads>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        Kernel_Full_4DTensor.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
	      stride
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
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_input,
    int stride
) {
  // Definition of the mapping `input_sequence_position -> output_sequence_position` when stride is considered
  /*
  Define output_position as the position of output when the first (or second) input is at input_sequence_position and the kernel multiplied is at kernel_position.

  For the first point, 
  input_sequence_position - kernel_position = output_position * stride

  output_position := (input_sequence_position - kernel_position) / stride

  For the second point,
  REAL_input_sequence_position = input_sequence_position - 1

  output_position := (REAL_input_sequence_position - kernel_position) / stride = ((input_sequence_position - 1) - kernel_position) / stride
  */

  // The range of kernel_position when stride is considered
  /*
  For the first point, 
  0 <= output_position <= output_sequence_length - 1

    (<=>)
  0 <= (input_sequence_position - kernel_position) / stride <= output_sequence_length - 1

    (<=>)
  kernel_position <= input_sequence_position and input_sequence_position - (output_sequence_length - 1) * stride <= kernel_position

  Given that (input_sequence_position % stride) <= kernel_position <= kernel_length - 1,

  max(input_sequence_position % stride, input_sequence_position - (output_sequence_length - 1) * stride) <= kernel_position <= min(kernel_length - 1, input_sequence_position)

  For the second point, 
  0 <= output_position <= output_sequence_length - 1

    (<=>)
  0 <= ((input_sequence_position - 1) - kernel_position) / stride <= output_sequence_length - 1

    (<=>)
  kernel_position <= input_sequence_position - 1 and (input_sequence_position - 1) - (output_sequence_length - 1) * stride <= kernel_position

  Given that (input_sequence_position - 1) % stride <= kernel_position <= kernel_length - 1,

  max((input_sequence_position - 1) % stride, (input_sequence_position - 1) - (output_sequence_length - 1) * stride) <= kernel_position <= min(kernel_length - 1, input_sequence_position - 1)
  */

  // get sizes
  const auto batch_size = input.size(0);
  const auto channel_size = input.size(1);
  const auto input_sequence_length = input.size(2);
  const auto kernel_length = Kernel_Full_4DTensor.size(0);
  const auto kernel_number = Kernel_Full_4DTensor.size(3);
  const auto output_sequence_length = grad_output.size(2);
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

      // clear the grad first
      grad_input[batch_index][channel_index][input_sequence_position] = 0;
      /* 
	 First point (valid only if input_sequence_position + 1 <= input_sequence_length - 1):
	 channel_1_index = channel_index as first point
	 iterate over the following:
	 temp_channel_2_index as second point \in 0, ..., channel_size - 1
	 temp_kernel_index \in 0, ..., kernel_number - 1
	 temp_kernel_position \in max(input_sequence_position % stride, input_sequence_position - (output_sequence_length - 1) * stride), ..., min(kernel_length - 1, input_sequence_position)
	 temp_output_sequence_position = (input_sequence_position - temp_kernel_position) / stride;
	 temp_grad_input_first_point += input[batch_index][temp_channel_2_index][input_sequence_position + 1] * Kernel_Full_4DTensor[temp_kernel_position][channel_1_index][temp_channel_2_index][temp_kernel_index] * grad_output[batch_index][temp_kernel_index][temp_output_sequence_position];
	 Final update: grad_input[batch_index][channel_1_index][input_sequence_position] += temp_grad_input_first_point;
      */
      scalar_t temp_grad_input_first_point = 0;
      const int channel_1_index = channel_index;
      if (input_sequence_position + 1 <= input_sequence_length - 1){
	for (int temp_channel_2_index=0; temp_channel_2_index < channel_size; temp_channel_2_index++){	      
	  for (int temp_kernel_index=0; temp_kernel_index < kernel_number; temp_kernel_index++){
	    for (int temp_kernel_position=max(input_sequence_position % stride, input_sequence_position - (output_sequence_length - 1) * stride); 
		 temp_kernel_position <= min(kernel_length - 1, input_sequence_position); 
		 temp_kernel_position+=stride){
	      int temp_output_sequence_position = (input_sequence_position - temp_kernel_position) / stride;
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
	 temp_kernel_position \in max((input_sequence_position - 1) % stride, (input_sequence_position - 1) - (output_sequence_length - 1) * stride), ..., min(kernel_length - 1, input_sequence_position - 1)
	 temp_output_sequence_position = (( input_sequence_position - 1 ) - temp_kernel_position ) / stride;
	 temp_grad_input_second_point += input[batch_index][temp_channel_1_index][input_sequence_position - 1] * Kernel_Full_4DTensor[temp_kernel_position][temp_channel_1_index][channel_2_index][temp_kernel_index] * grad_output[batch_index][temp_kernel_index][temp_output_sequence_position];
	 Final update: grad_input[batch_index][channel_2_index][input_sequence_position] += temp_grad_input_second_point;
      */
      scalar_t temp_grad_input_second_point = 0;
      const int channel_2_index = channel_index;
      if (input_sequence_position >= 1){
	for (int temp_channel_1_index=0; temp_channel_1_index < channel_size; temp_channel_1_index++){	      
	  for (int temp_kernel_index=0; temp_kernel_index < kernel_number; temp_kernel_index++){
	    for (int temp_kernel_position=max((input_sequence_position - 1) % stride, (input_sequence_position - 1) - (output_sequence_length - 1) * stride);
		 temp_kernel_position <= min(kernel_length - 1, input_sequence_position - 1);
		 temp_kernel_position+=stride){
	      int temp_output_sequence_position = ((input_sequence_position - 1) - temp_kernel_position) / stride;
	      temp_grad_input_second_point += input[batch_index][temp_channel_1_index][input_sequence_position - 1] * Kernel_Full_4DTensor[temp_kernel_position][temp_channel_1_index][channel_2_index][temp_kernel_index] * grad_output[batch_index][temp_kernel_index][temp_output_sequence_position];
	    }
	  }
	}
	grad_input[batch_index][channel_2_index][input_sequence_position] += temp_grad_input_second_point;
      }

      // Finish

  }
}



template <typename scalar_t>
__global__ void markonv_cuda_backward_grad_Kernel_Full_4DTensor_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> Kernel_Full_4DTensor,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_Kernel_Full_4DTensor,
    int stride
								       ) {
  // Definition of the mapping `input_sequence_position -> output_sequence_position` when stride is considered
  /*
  FIRST_input_sequence_position = output_sequence_position * stride + kernel_position
  SECOND_input_sequence_position = output_sequence_position * stride + 1 + kernel_position
  */
  // get sizes
  const auto batch_size = input.size(0);
  const auto channel_size = input.size(1);
  const auto input_sequence_length = input.size(2);
  const auto kernel_length = Kernel_Full_4DTensor.size(0);
  const auto kernel_number = Kernel_Full_4DTensor.size(3);
  const auto output_sequence_length = grad_output.size(2);
  // mixed_index
  const int mixed_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (mixed_index < kernel_number * channel_size * channel_size * kernel_length){
    //  
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
	int temp_input_sequence_position = temp_output_sequence_position * stride;
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
    torch::Tensor Kernel_Full_4DTensor /* shape: [kernel_length, channel, channel, kernel_number] */,
    int stride) {
  
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
        grad_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
	stride
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
        grad_Kernel_Full_4DTensor.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
	stride
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
