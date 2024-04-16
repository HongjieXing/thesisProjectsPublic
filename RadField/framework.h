#pragma once

#define WIN32_LEAN_AND_MEAN             // 从 Windows 头文件中排除极少使用的内容
// Windows 头文件
#include <windows.h>
#include<iostream>
#include<cmath>
#include<iomanip>
#include<fstream>
#include<sstream>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>
#include<string>
#include<ctime>
#include<thread>
#include <algorithm>
#include <cassert>

#define mainMode 2
// width = 2 mm / pixel
#define ROWS 192
#define COLS 192
#define numThreadMode 0 //0:auto or set; n>0: n(fixed); n<0: hardware_concurrency()

/*
control the lamp position associate with the receiver.
0:auto, lamp positions modify with the domain
1:fixed, search basicLength_x to set the adjacent lamp space
*/
#define domainMode 1 //0:auto(modify with the domain), 1:fixed,limited in the domain
//#define NOEBUG