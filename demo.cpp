#include "bitonic_sort.h"
#include <cstdio>
#include <sys/mman.h>
#include <stdlib.h> 
#include <cassert>

using namespace std;

void print(int *tab, int n){
  for (int i = 0; i < n; ++i) {
    printf("%d \n", tab[i]);
  }
}

int main(){
   int n = 1024*1024;
   int *c = (int*) malloc(n*sizeof(int));
   for (int j=0; j<n; ++j){
       c[j] = n-j;
   }
   int* d = bitonic_sort(c, n);
   for (int j=0; j<(n-1); ++j){
      if (d[j] > d[j + 1]) {
        printf("%d %d\n", d[j], d[j+1]);
      } 
      assert(d[j] <= d[j + 1] );
      if (d[j] +1 != d[j + 1]) printf("%d %d\n", d[j], d[j+1]);
      assert(d[j] +1 == d[j + 1]);
     
  }
   printf("ok\n");
   free(c);
   free(d);
   return 0;
}