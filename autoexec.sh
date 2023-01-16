#!/bin/zsh



echo "im1"
python3 main.py 1080p.jpg 1 > tmp
var=$(sed -n "2p" "tmp" | cut -d ' ' -f4) 
echo $var";" >> res
var1=$(sed -n "3p" "tmp" | cut -d ' ' -f6)
echo $var1";" >> res1 

python3 main.py 1080p.jpg 2 > tmp
var=$(sed -n "2p" "tmp" | cut -d ' ' -f4) 
echo $var";" >> res
var1=$(sed -n "3p" "tmp" | cut -d ' ' -f6)
echo $var1";" >> res1 

python3 main.py 1080p.jpg 3 > tmp
var=$(sed -n "2p" "tmp" | cut -d ' ' -f4) 
echo $var";" >> res
var1=$(sed -n "3p" "tmp" | cut -d ' ' -f6)
echo $var1";" >> res1 

python3 main.py 1080p.jpg 4 > tmp
var=$(sed -n "2p" "tmp" | cut -d ' ' -f4) 
echo $var";" >> res
var1=$(sed -n "3p" "tmp" | cut -d ' ' -f6)
echo $var1";" >> res1 

python3 main.py 1080p.jpg 5 > tmp
var=$(sed -n "2p" "tmp" | cut -d ' ' -f4) 
echo $var";" >> res
var1=$(sed -n "3p" "tmp" | cut -d ' ' -f6)
echo $var1";" >> res1 


python3 main.py 1080p.jpg 6 > tmp
var=$(sed -n "2p" "tmp" | cut -d ' ' -f4) 
echo $var";" >> res
var1=$(sed -n "3p" "tmp" | cut -d ' ' -f6)
echo $var1";" >> res1 

python3 main.py 1080p.jpg 7 > tmp
var=$(sed -n "2p" "tmp" | cut -d ' ' -f4) 
echo $var";" >> res
var1=$(sed -n "3p" "tmp" | cut -d ' ' -f6)
echo $var1";" >> res1 


# echo "im1"
# python3 main.py 480p.jpg 6 > tmp
# var=$(sed -n "3p" "tmp" | cut -d ' ' -f4) 
# echo $var";" >> res
# var1=$(sed -n "4p" "tmp" | cut -d ' ' -f6)
# echo $var1";" >> res1 


# echo "im2"
# python3 main.py 1080p.jpg 6 > tmp
# var=$(sed -n "3p" "tmp" | cut -d ' ' -f6) 
# echo $var";" >> res
# var1=$(sed -n "5p" "tmp" | cut -d '=' -f2)
# echo $var1";" >> res1 


# echo "im3"
# python3 main.py 2K.jpg 6 > tmp
# var=$(sed -n "3p" "tmp" | cut -d ' ' -f6) 
# echo $var";" >> res
# var1=$(sed -n "5p" "tmp" | cut -d '=' -f2)
# echo $var1";" >> res1 

# echo "im4"
# python3 main.py 4K.jpg 6 > tmp
# var=$(sed -n "3p" "tmp" | cut -d ' ' -f6) 
# echo $var";" >> res
# var1=$(sed -n "5p" "tmp" | cut -d '=' -f2)
# echo $var1";" >> res1 

# echo "im5"
# python3 main.py 8K.jpg 6 > tmp
# var=$(sed -n "3p" "tmp" | cut -d ' ' -f6) 
# echo $var";" >> res
# var1=$(sed -n "5p" "tmp" | cut -d '=' -f2)
# echo $var1";" >> res1 




