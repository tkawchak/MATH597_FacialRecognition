#!/bin/bash

# take a directory in the form
# 1
#     aaa
#     bbb
# 2
#     aaa
# 3
#     aaa
#     bbb
#     ccc

# make a directory
# 1_1
# 2_1
# 3_2
# 4_3
# 5_3
# 6_3

#find . -mindepth 2 -type f -print -exec mv {} . \;
cd sampleData;
count=0;
extension=".jpg"
for d in * ; do
    cd $d;
    for f in * ; do
        let count=count+1
        #echo "$count""_"$d$extension;
	mv "$f" "../""$count""_"$d$extension;
    done
    cd ../;
    rmdir $d
done
