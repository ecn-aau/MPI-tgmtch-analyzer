#!/bin/bash
LD_LIBRARY_PATH="/home/es.aau.dk/az05mg/.local/lib"
export LD_LIBRARY_PATH

rm -rf ./dumps/*
rm -rf ./pictures/

mkdir -p ./dumps/
mkdir -p ./pictures/

sizes=(1 16 32 64 128 256)

cargo b --release

# Print DUMPI stuff
echo
echo "DUMPI2ASCII information (important)"

echo LD_LIBRARY_PATH: $LD_LIBRARY_PATH
echo LDD info: 
ldd $(which dumpi2ascii)

echo
echo

for dir in "${1}"*/ ; do
    [ -L "${d%/}" ] && continue

    echo Simulating "$dir"

    rm -f ${dir}mts.cache
    
    # Get name.txt file for "sanitized" name
    name_file="${dir}name.txt"    
    if [ ! -f $name_file ]; then
        echo "name.txt not found! Skipping, no sanitized name for application!"
        echo
        continue
    fi
    
    app_name=$(cat $name_file)
    dir_name=$(basename "$app_name")

    mkdir -p "./dumps/${dir_name}"
    
    for s in ${sizes[@]}; do
        output="./dumps/${dir_name}/df${s}"
        cargo r --release -- -t $dir --nb $s --sb $s --tb $s --missing -o $output -g 1
    done
    echo
done

wait