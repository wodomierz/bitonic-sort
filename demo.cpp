#include "bitonic_sort.h"
#include <cstdio>
#include <cstdlib> 
#include <cassert>
#include <climits> 
#include <sys/mman.h>

using namespace std;

void test0();
void test1();
void test2();
void test3();
void test_big();

void print(int *tab, int n){
  for (int i = 0; i < n; ++i) {
    printf("%d \n", tab[i]);
  }
}

int main(){
  test0();
  test1();
  test2();
  test3();
  test_big();
  return 0;
}

void test0() {
  int n = 1024*1024;
  int *c = (int*) malloc(n*sizeof(int));
  for (int j=0; j<n; ++j) {
      c[j] = n-j;
  }
  int* d = bitonic_sort(c, n);
  for (int j=0; j<(n-1); ++j) {
    if (d[j] > d[j + 1]) {
      printf("test0 %d %d\n", d[j], d[j+1]);
    } 
    assert(d[j] <= d[j + 1]);
    if (d[j] +1 != d[j + 1]) {
      printf("test0 %d %d\n", d[j], d[j+1]);
    }
    assert(d[j] +1 == d[j + 1]);
  }
  printf("test0 ok\n");
  free(c);
  free(d);
}

void test1() {
  int n = 1024*1024;
  int *c = (int*) malloc(n*sizeof(int));
  for (int j=0; j<n; ++j){
      c[j] = rand();
  }
  int* d = bitonic_sort(c, n);
  for (int j=0; j<(n-1); ++j) {
      if (d[j] > d[j + 1]) {
        printf("test1 %d %d\n", d[j], d[j+1]);
      } 
      assert(d[j] <= d[j + 1]);
  }
  printf("test1 ok\n");
  free(c);
  free(d);
}


void test2() {
  int n = 1024 * 23;
  int *c = (int*) malloc(n*sizeof(int));
  for (int j=0; j<n; ++j){
      c[j] = rand();
  }
  int* d = bitonic_sort(c, n);
  for (int j=0; j<(n-1); ++j) {
      if (d[j] > d[j + 1]) {
        printf("test2 %d %d\n", d[j], d[j+1]);
      } 
      assert(d[j] <= d[j + 1]);
  }
  printf("test2 ok\n");
  free(c);
  free(d);
}

void test3() {
  int n = 10899;
  int *c = (int*) malloc(n*sizeof(int));
  for (int j=0; j<n; ++j){
      c[j] = rand();
  }
  int* d = bitonic_sort(c, n);
  for (int j=0; j<(n-1); ++j) {
      if (d[j] > d[j + 1]) {
        printf("test3 %d %d\n", d[j], d[j+1]);
      } 
      assert(d[j] <= d[j + 1]);
  }
  printf("test3 ok\n");
  free(c);
  free(d);
}

void test_big() {
  int times = 1;
  int min = 1023;
  int max = 1024*1024;
  while (times++ < 50) {
    int n = min + (rand() % (int)(max - min + 1));
    int *c = (int*) malloc(n*sizeof(int));
    for (int j=0; j<n; ++j) {
      c[j] = rand();
    }
    int* d = bitonic_sort(c, n);
    for (int j=0; j<(n-1); ++j) {
      if (d[j] > d[j + 1]) {
        printf("testbig%d %d %d\n", times, d[j], d[j+1]);
      } 
      assert(d[j] <= d[j + 1]);
    }
    free(c);
    free(d);
  }  
  printf("testbig ok\n");
}