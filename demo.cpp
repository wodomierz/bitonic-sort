#include "bitonic_sort.h"
#include <cstdio>
#include <sys/mman.h>
#include <stdlib.h> 
#include <cassert>

using namespace std;

int main(){
   int n = 1024*2;
   int *c = (int*) malloc(n*sizeof(int));
   for (int j=0; j<n; ++j){
       c[j] = n-j;
   }
   int* d = bitonic_sort(c, n);
   for (int j=0; j<n; ++j){
      printf("%d %d\n", d[j], d[j+1]);
      // assert(d[j] == c[n - 1 -j]);
       // assert(d[j] <= d[j + 1] );
  }
   printf("ok\n");
   free(c);
   free(d);
   return 0;
}