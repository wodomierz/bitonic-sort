#include "bitonic_sort.h"
#include <cstdio>
#include <cstdlib> 
#include <cassert>
#include <climits> 
#include <sys/mman.h>

#include <ctime> 
#include <cstring>
#include <algorithm>
#include <iostream>

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

void comparesorts(int n) {
  // int n = 1024*1024*50;
  cout << "Size "<< n<< ":"<<endl;

  int *c1 = (int*) malloc(n*sizeof(int));
  int *c2 = (int*) malloc(n*sizeof(int));
  std::clock_t start;
  int* d;

  //rand
  for (int j=0; j<n; ++j) {
      c1[j] = rand();
  }
  memcpy(c2, c1, n*sizeof(int));
  start = std::clock();
  d = bitonic_sort(c1, n);
  std::cout << "Time for bitonic: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
  start = std::clock();
  sort(c2, c2 + n);
  std::cout << "Time for stl: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << endl;
  free(d);

  // ascending
  for (int j=0; j<n; ++j) {
      c1[j] = j;
  }
  memcpy(c2, c1, n*sizeof(int));
  start = std::clock();
  d = bitonic_sort(c1, n);
  std::cout << "Time for bitonic asc: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
  start = std::clock();
  sort(c2, c2 + n);
  std::cout << "Time for stl asc: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << endl;
  free(d);

  //descending
  for (int j=0; j<n; ++j) {
      c1[j] = n-j;
  }
  memcpy(c2, c1, n*sizeof(int));
  start = std::clock();
  d = bitonic_sort(c1, n);
  std::cout << "Time for bitonic desc: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

  start = std::clock();
  sort(c2, c2 + n);
  std::cout << "Time for stl desc: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << endl;
  free(d);

  free(c1);
  free(c2);
}

int main(){
  srand( time( NULL ) );
  cout << "===================" <<endl; 
  cout << "multiples of 1024:" << endl;
  cout << "===================" <<endl; 
  cout << endl;

  for (int i = 1; i < 1024*64; i *= 2) {
    comparesorts(i*1024);
  }

  cout << endl;
  cout << "===================" <<endl; 
  cout << "not multiples of 1024:" << endl;
  cout << "===================" <<endl; 
  cout << endl;

  for (int i = 1; i < 1024*64; i *= 2) {
    comparesorts(i*1025);
  }
  
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