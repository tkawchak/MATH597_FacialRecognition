#!/bin/bash

# all tightly cropped images
curl megaface.cs.washington.edu/dataset/download/content/identities_tight_cropped.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_tight_cropped.tar.gz
tar -xvf identities_tight_cropped.tar.gz
rm -f identities_tight_cropped.tar.gz

# all lossely cropped images, done by 14 groups
#curl megaface.cs.washington.edu/dataset/download/content/identities_0.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_0.gz
mv identities_0.gz identities_0.tar.gz
tar -xvf identities_0.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_0.tar.gz

#curl megaface.cs.washington.edu/dataset/download/content/identities_1.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_1.gz
mv identities_1.gz identities_1.tar.gz
tar -xvf identities_1.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_1.tar.gz

#curl megaface.cs.washington.edu/dataset/download/content/identities_2.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_2.gz
mv identities_2.gz identities_2.tar.gz
tar -xvf identities_2.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_2.tar.gz

#curl megaface.cs.washington.edu/dataset/download/content/identities_3.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_3.gz
mv identities_3.gz identities_3.tar.gz
tar -xvf identities_3.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_3.tar.gz

curl megaface.cs.washington.edu/dataset/download/content/identities_4.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_4.tar.gz
tar -xvf identities_4.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_4.tar.gz

curl megaface.cs.washington.edu/dataset/download/content/identities_5.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_5.tar.gz
tar -xvf identities_5.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_5.tar.gz

curl megaface.cs.washington.edu/dataset/download/content/identities_6.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_6.tar.gz
tar -xvf identities_6.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_6.tar.gz

curl megaface.cs.washington.edu/dataset/download/content/identities_7.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_7.tar.gz
tar -xvf identities_7.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_7.tar.gz

curl megaface.cs.washington.edu/dataset/download/content/identities_8.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_8.tar.gz
tar -xvf identities_8.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_8.tar.gz

curl megaface.cs.washington.edu/dataset/download/content/identities_9.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_9.tar.gz
tar -xvf identities_9.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_9.tar.gz

curl megaface.cs.washington.edu/dataset/download/content/identities_10.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_10.tar.gz
tar -xvf identities_10.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_10.tar.gz

curl megaface.cs.washington.edu/dataset/download/content/identities_11.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_11.tar.gz
tar -xvf identities_11.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_11.tar.gz

curl megaface.cs.washington.edu/dataset/download/content/identities_12.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_12.tar.gz
tar -xvf identities_12.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_12.tar.gz

curl megaface.cs.washington.edu/dataset/download/content/identities_13.tar.gz --user tyk5178@psu.edu:3Avj/X.o7z --verbose --output identities_13.tar.gz
tar -xvf identities_13.tar.gz --directory MegafaceIdentities_Uncropped
rm -f identities_13.tar.gz