#include <stdio.h>

unsigned char* readBMP(char* filename){

    int i,j;
    FILE* f=fopen(filename,"rb");

    if(f==NULL) throw "Argument Exception";

    unsigned char info[54];
    fread(info,sizeof(unsigned char),54,f);

    int width=*(int*)&info[18];
    int height=*(int*)&info[22];

    int size=3*width*height;
    unsigned char* data=new unsigned char[size];
    fread(data,sizeof(unsigned char),size,f);
    fclose(f);

    unsigned char RedMatrix [224][224];
    unsigned char GreenMatrix [224][224];
    unsigned char BlueMatrix [224][224];

    for(i=0;i<224;i++){
        for(j=0;j<224;j++){
            if(i>sizeOf(data)/sizeOf(unsigned char)||j>sizeOf(data)/sizeOf(unsigned char)){
                RedMatrix[i][j]=0;
                GreenMatrix[i][j]=0;
                BlueMatrix[i][j]=0;
            }
            else{
                RedMatrix[i][j]=data[j*width+i+2];
                GreenMatrix[i][j]=data[j*width+i+1];
                BlueMatrix[i][j]=data[j*width+i];
            }
        }
    }
    unsigned char pixels [3]={RedMatrix,GreenMatrix,BlueMatrix};
    return pixels;
}
