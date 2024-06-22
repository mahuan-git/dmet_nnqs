#!/bin/bash

# Check if the user has provided a filename as argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Filename
filename=$1

# Use sed for the replacement
sed -i 's/^.*use_restricted_hilbert: true.*$/electron_conservation_type: "restricted"/' $filename
sed -i '/use_restricted_hilbert/d' $filename

