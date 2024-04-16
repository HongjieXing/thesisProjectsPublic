// pch.h: 这是预编译标头文件。
// 下方列出的文件仅编译一次，提高了将来生成的生成性能。
// 这还将影响 IntelliSense 性能，包括代码完成和许多代码浏览功能。
// 但是，如果此处列出的文件中的任何一个在生成之间有更新，它们全部都将被重新编译。
// 请勿在此处添加要频繁更新的文件，这将使得性能优势无效。

#ifndef PCH_H
#define PCH_H

// 添加要在此处预编译的标头
#include "framework.h"
#include "Heater.h"
#include "Photon.h"
#include "Field.h"
#include "BasicClasses.h"

// 下列 ifdef 块是创建使从 DLL 导出更简单的宏的标准方法。
// 在预处理器中定义了 MYDLL_EXPORTS 符号，而调用方不包含此符号。
// 源文件中包含此文件的任何其他项目都会将 MYDLL_API 视为是从 DLL 导入的，
// 而此 DLL 则将用此宏定义的符号视为是被导出的。
#ifdef _DLLAPI
	#define DLLAPI __declspec(dllexport)
#else
	#define DLLAPI __declspec(dllimport)
#endif


#endif //PCH_H