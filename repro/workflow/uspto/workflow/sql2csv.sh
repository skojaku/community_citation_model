#!/bin/bash
length=$(($#-1))
input_files=${@:1:$length}
output_file="${@: -1}"

if [ -f $output_file ] ; then
    rm $output_file 
fi

for input_file in $input_files 
do
  grep "^(" $input_file |sed -e "s/(\|)\|);//g"|sed -e "s/,$//g" >>$output_file
done
