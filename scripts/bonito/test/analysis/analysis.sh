#!/usr/bin/env bash

# Copyright 2019 Ryan Wick (rrwick@gmail.com)
# https://github.com/rrwick/Basecalling-comparison

# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You
# should have received a copy of the GNU General Public License along with this program. If not,
# see <http://www.gnu.org/licenses/>.

# This script:
#   * takes one argument: a fastq/fasta file of reads for analysis:
#      * read names should be in UUID format
#      * reads should be sorted by name
#   * expects the following in the directory in which it's run:
#      * a reference.fasta file
#   * expects the following tools in the PATH:
#      * minimap2 (https://github.com/lh3/minimap2)
#      * rebaler (https://github.com/rrwick/Rebaler)
#      * racon (https://github.com/isovic/racon)
#      * nanopolish and nanopolish_makerange.py (https://github.com/jts/nanopolish)
#      * MUMmer tools: nucmer, delta-filter, show-snps (https://sourceforge.net/projects/mummer)


# Some high-level settings for the script:
threads=20
assembly_dir=../../../result/bonito/assemblies     # where finished assemblies will go
nanopolish_dir=../../../result/bonito/nanopolish   # where Nanopolished assemblies will go
results_dir=../../../result/bonito         # where the result tables will go


# This variable holds the directory of the other scripts (assumed to be in the same directory as
# this script):
scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


# Get the read set name from the filename:
set=$(basename $1 | sed 's/.fastq.gz//' | sed 's/.fasta.gz//' | sed 's/.fastq//' | sed 's/.fasta//')
if [[ $1 = *"fastq"* ]]; then read_type=fastq
elif [[ $1 = *"fasta"* ]]; then read_type=fasta
else echo "Error: cannot determine read type"; exit 1
fi


# Various file paths that will be used along the way:
read_data="$results_dir"/"$set"_reads.tsv
read_alignment="$results_dir"/"$set"_reads.paf
trimmed_reads="$assembly_dir"/"$set"_trimmed."$read_type"
filtlong_reads="$assembly_dir"/"$set"_filtlong."$read_type"
assembly_reads="$assembly_dir"/"$set"_assembly_reads.fastq
final_assembly="$assembly_dir"/"$set"_assembly.fasta
assembly_pieces="$results_dir"/"$set"_assembly_pieces.fasta
assembly_data="$results_dir"/"$set"_assembly.tsv
assembly_alignment="$results_dir"/"$set"_assembly.paf
nanopolish_assembly="$nanopolish_dir"/"$set"_nanopolish.fasta
sequencing_summary=03_basecalling_output/"$set"/sequencing_summary.txt


# Used for nucmer:
prefix="$set"_details
ref_contig=chromosome
assembly_contig=chromosome



# 1. evaluate read accuracy
printf "\n\n\n"
echo "ASSESS READS: "$set
echo "--------------------------------------------------------------------------------"
mkdir -p "$results_dir"
minimap2 -x map-ont -t $threads -c reference.fasta $1 > $read_alignment
python3 "$scripts_dir"/read_length_identity.py $1 $read_alignment > $read_data
rm $read_alignment


# 2. evaluate consensus accuracy
printf "\n\n\n\n"
echo "ASSEMBLY: "$set
echo "--------------------------------------------------------------------------------"
mkdir -p "$assembly_dir"
# Run Porechop
#   --no_split: to save time, chimeras shouldn't matter for Rebaler anyway
#   --check_reads 1000: to save time, it's all one barcode, so 1000 reads should be plenty
porechop -i $1 -o $trimmed_reads --no_split --threads $threads --check_reads 1000
if [ "$read_type" = "fasta" ]; then
    seqtk seq $trimmed_reads > "$assembly_dir"/temp.fasta  # make one-line-per-seq fasta
    mv "$assembly_dir"/temp.fasta $trimmed_reads
fi
# Run multiple replicates of Racon on shuffled reads and a rotated reference.
# Each assembly is then coursely shredded into 'reads' for the final Racon round.
for i in {0..9}; do
    sample_reads="$assembly_dir"/"$set"_sample_"$i"."$read_type"
    sample_assembly="$assembly_dir"/"$set"_assembly_"$i".fasta
    sample_reference="$assembly_dir"/"$set"_reference_"$i".fasta
    if [ "$read_type" = "fastq" ]; then
        cat $trimmed_reads | paste - - - - | shuf | awk -F'\t' '{OFS="\n"; print $1,$2,$3,$4}' > $sample_reads
    elif [ "$read_type" = "fasta" ]; then
        cat $trimmed_reads | paste - - | shuf | awk -F'\t' '{OFS="\n"; print $1,$2}' > $sample_reads
    fi
    python3 "$scripts_dir"/rotate_reference.py reference.fasta $i > $sample_reference
    rebaler -t $threads $sample_reference $sample_reads > $sample_assembly
    python3 "$scripts_dir"/shred_assembly.py $sample_assembly $i 50000 >> $assembly_reads
    rm $sample_reads $sample_assembly $sample_reference
done
rm $trimmed_reads
rebaler -t $threads reference.fasta $assembly_reads > $final_assembly
rm $assembly_reads



printf "\n\n\n\n"
echo "ASSESS ASSEMBLY: "$set
echo "--------------------------------------------------------------------------------"
python3 "$scripts_dir"/chop_up_assembly.py $final_assembly 10000 > $assembly_pieces
minimap2 -x asm5 -t $threads -c reference.fasta $assembly_pieces > $assembly_alignment
python3 "$scripts_dir"/read_length_identity.py $assembly_pieces $assembly_alignment > $assembly_data
rm $assembly_pieces $assembly_alignment
if [ ! -f "$results_dir"/assembly_error_details ]; then
    printf "assembly\tdcm\thomo_del\thomo_ins\tother_del\tother_ins\tsub\n" > "$results_dir"/assembly_error_details
fi
printf $set"\t" >> "$results_dir"/assembly_error_details
nucmer --prefix="$prefix" reference.fasta $final_assembly
delta-filter -r -q "$prefix".delta > "$prefix".filter
show-snps -ClrTH -x5 "$prefix".filter | python3 "$scripts_dir"/error_summary.py "$ref_contig" "$assembly_contig" >> "$results_dir"/assembly_error_details
printf $set"\tassembly\t" >> "$results_dir"/substitution_counts
show-snps -ClrTH "$prefix".filter | awk '$2 != "." && $3 != "."' | wc -l >> "$results_dir"/substitution_counts
rm "$prefix".delta "$prefix".filter



