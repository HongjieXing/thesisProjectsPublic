#include"../pch.h"

extern "C"//�����C���Եĺ���
{
	extern DLLAPI double* LDF = nullptr;
	extern DLLAPI double* LPF = nullptr;
	DLLAPI PlateSystem* Heater;
	// ָ��
	extern DLLAPI int row = ROWS;
	extern DLLAPI int col = COLS;
	
	// * * * * * * * * * * * * * * * * * *
	DLLAPI void fieldConstructor();
	DLLAPI void createHeater(const char* info, int numLamps);
	DLLAPI bool runLDFandLPF();
	DLLAPI void deleteField();
	DLLAPI void printField(const char* info);
}

bool calcuLPF();
bool calcuLDF();