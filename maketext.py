import sys,os;
rootdir="/media/seongwon/U"
f = open("val.txt", 'w')
li=os.listdir(rootdir)
for file in li:
    if file != "LICENSE" :
        f.write(file)
        f.write("\n")

f.close()