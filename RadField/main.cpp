#include"../RadFieldDLL/pch.h"
#pragma comment(lib, "RadFieldDLL.lib") //需要在:F\RadField\testMain\main.cpp目录拷贝上新生成的lib文件
#if mainMode == 1

#include <iostream>
#include <algorithm>
#include <map>
void runRadLDF(unsigned int sampleNumber);
void runRadLDFTestMode(unsigned int sampleNumber);
void run3DRadLDF(unsigned int sampleNumber);
void runMultiLayersRadLDF(unsigned int sampleNumber, int numLamps);
#define includeLPF 1
#define runMode 1 //train 0, train 1(test: 从train中分出一部分就行）, 3D 2, multiLayers 3


int main() {

	srand((unsigned int)time(NULL));
	const int MaxSampleNum = 5000;

	{
#if numThreadMode == 0
		std::cout << "Threads number is given in main() ..." << std::endl;
		
#elif numThreadMode < 0
		std::cout << "Using hardware concurrency automatically..." << std::endl;
		std::cout << "Hardware concurrency in this machine = " << std::thread::hardware_concurrency() << std::endl;
#else
		std::cout << "Warning, using " << numThreadMode << " threads for Rad calculation." << std::endl;
#endif

		std::cout << "The size(row * col) of output matrix = " << int(ROWS) << " * " << int(COLS) << std::endl;
#if domainMode == 1
		std::cout << "The lamps are limited in the range of calculation domain..." << std::endl;
#elif domainMode == 0
		std::cout << "The lamps are modify with the domain range automaticly..." << std::endl;
#endif
	}

#if runMode == 0 // train, include used samples

	std::ifstream usedSample;
	usedSample.open("./log/logSampleNumber.txt", std::ios::app | std::ios::out);
	if (!usedSample) {
		std::cout << "Error, cannot open logSampleNumber file.\n";
	}
	int sumUsedSample = 1; //depend on the input file
	std::cout << "In file './log/logSampleNumber.txt', Number of used samples = ";
	std::cin >> sumUsedSample;
	std::map<int, int>mapUsedSample;
	for (int i = 0; i < sumUsedSample; i++) {
		int tmp;
		usedSample >> tmp;
		mapUsedSample[tmp];
	}
	usedSample.close();

	/*std::ofstream LOG;
	LOG.open("./log/logSampleNumberAdd.txt", std::ios::app | std::ios::out);
	if (!LOG) {
		std::cout << "Error, cannot open log file.\n";
	}
	const int sumSample = 250;
	std::map<int, int>mapSample;
	for (int i = 0; i < sumSample; i++) {
		int tmpSample = rand() % MaxSampleNum;
		auto t1 = mapSample.find(tmpSample);
		auto t2 = mapUsedSample.find(tmpSample);
		if (t1 == mapSample.end() && t2 == mapUsedSample.end()) {
			mapSample[tmpSample];
		}
		else {
			i--;
		}
	}

	for (auto it = mapSample.begin(); it != mapSample.end(); it++) {
		LOG << it->first << std::endl;
	}
	LOG << std::endl;
	LOG.close();*/

	double Progress = 0;
	for (auto it = mapUsedSample.begin(); it != mapUsedSample.end(); it++) {
		std::cout << "\nProgress = " << Progress << "%\n";
		runRadLDF(it->first);
		Progress += 1.0 / double(sumUsedSample) * 100.0;
	}
	mapUsedSample.clear();
#endif

#if runMode == 1 // train, generate brand new samples
	std::ofstream LOG;
	LOG.open("./log/logTrainSampleNumber.txt", std::ios::app | std::ios::out);
	if (!LOG) {
		std::cout << "Error, cannot open logTestSampleNumber file.\n";
	}
	const int sumSample = 80; //test sample number
	std::map<int, int>mapSample;
	for (int i = 0; i < sumSample; i++) {
		int tmp = rand() % MaxSampleNum;
		auto t1 = mapSample.find(tmp);
		if (t1 == mapSample.end()) 
		{
			mapSample[tmp];
		}
		else {
			i--;
		}
	}
	for (auto it = mapSample.begin(); it != mapSample.end(); it++) {
		LOG << it->first << std::endl;
	}
	LOG.close();
	

	double Progress = 0;
	for (auto it = mapSample.begin(); it != mapSample.end(); it++) {
		std::cout << "\nProgress = " << Progress << "%\n";
		runRadLDF(it->first);
		Progress += 1.0 / double(sumSample) * 100.0;
	}
	mapSample.clear();
#endif

#if runMode == 2 // 3D increase receiverY
	
	run3DRadLDF(0); //sampleNumber = 0000, lamps = 6, layer = 1
	
#endif

#if runMode == 3 // 多层
	/*std::ifstream usedSample;
	usedSample.open("./log/logSampleNumber.txt", std::ios::app | std::ios::out);
	if (!usedSample) {
		std::cout << "Error, cannot open logTestSampleNumber file.\n";
	}*/
	std::ofstream LOG;
	LOG.open("./log/logTestSampleNumber.txt", std::ios::app | std::ios::out);
	if (!LOG) {
		std::cout << "Error, cannot open logTestSampleNumber file.\n";
	}
	const int sumSample = 100; //test sample number
	//const int sumUsedSample = 2000; //depend on the input file
	std::map<int, int>mapSample;
	/*std::map<int, int>mapUsedSample;
	for (int i = 0; i < sumUsedSample; i++) {
		int tmp;
		usedSample >> tmp;
		mapUsedSample[tmp];
	}*/
	for (int i = 0; i < sumSample; i++) {
		int tmp = rand() % MaxSampleNum;
		while (/*mapUsedSample.find(tmp) != mapUsedSample.end() ||*/
			mapSample.find(tmp) != mapSample.end()) {
			tmp = rand() % MaxSampleNum;
		}
		mapSample[tmp];
	}
	for (auto it = mapSample.begin(); it != mapSample.end(); it++) {
		LOG << it->first << std::endl;
	}

	//usedSample.close();
	LOG << std::endl << "numLamps:" << std::endl;
	unsigned int numLampArr[5] = { 6, 10, 12, 16, 18 }; //注意，灯管数要成为单层的两倍！！！或者三倍
	double Progress = 0;
	/*for (auto it = mapSample.begin(); it != mapSample.end(); it++) {
		std::cout << "\nProgress = " << Progress << "%\n";
		int select = int(rand() % 5);
		Sleep(100);
		int numLamps = numLampArr[select];
		LOG << it->first << " " << numLamps << std::endl;
		runMultiLayersRadLDF(it->first, numLamps);
		Progress += 1.0 / double(sumSample) * 100.0;
	}*/
	for (int i = 7; i < 26; i+=2)
	{
		int numLamps = i;
		runMultiLayersRadLDF(0, numLamps);
	}
	LOG.close();
	mapSample.clear();
	//mapUsedSample.clear();
#endif


	system("pause");
	return 0;
}

/* * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * */


void runRadLDF(unsigned int sampleNumber) {
	//for (unsigned int numLamps = 3; numLamps < 10; numLamps++) //[3,10]
	//{
	//	if (numLamps == 4 || numLamps == 7) continue;//3 5 6 8 9
	//	PlateSystem plateSystem(numLamps);// Only set the number of lamps. Other parameters are default.
	//	plateSystem.generatePlateSystem(sampleNumber); // sampleNumber is a variable within [0, 4999]

	//	// Must use the same plateSystem, simultaneously compute
	//	LDFField ldfField(plateSystem);
	//	ldfField.runLDFField();
	//	
	//	RadField radField(plateSystem);
	//	radField.setPhotonNumber(4000);
	//	radField.runRadiantField(64);
	//}
	
	int numThreads = 12;
	// ----- 1 ----- 单层
	int numLamps1 = 3 + rand() % 10;
	
	/*for (auto it = numLamps1.begin(); it != numLamps1.end(); it++) 
	{*/
		PlateSystem plateSystem1(numLamps1);// Only set the number of lamps. Other parameters are default.
		plateSystem1.generatePlateSystem(sampleNumber); // sampleNumber is a variable within [0, 4999]

		// Must use the same plateSystem, simultaneously compute
		LDFField ldfField1(plateSystem1);
		ldfField1.runLDFField();
#if includeLPF == 1
		ldfField1.runLPFField();
#endif
		RadField radField1(plateSystem1);
		//radField1.setPhotonNumber(4000);
		radField1.runRadiantField(numThreads);
	//}

	double dy[6] = { 80.0, 160.0, 240.0, 320.0, 400.0, 480.0 };
	double ddy = 10.0;
	// ----- 2 ----- 多层
	for (int i = 0; i < 2; i++) 
	{
		unsigned int layers = 2 + rand() % 2;
		int numLamps2 = layers * 3 + rand() % 10 * layers;
		Sleep(10);
		//bool isStaggered = rand() % 2;
		bool isStaggered = 0; //设置为不交错
		std::stringstream info;
		info << layers << isStaggered;
		double deltaY = dy[rand() % 6] + ddy * (double(rand() % 7) - 3.0);
		if (i == 1)
		{			
			info << "_" << std::setw(3) << std::setfill('0') << int(deltaY);
		}
		PlateSystem plateSystem2(numLamps2);// Only set the number of lamps. Other parameters are default.
		double lampL = 360.0 / double(layers) + 5.0 * double(rand() % 10);
		double gaps[5] = { 50.0, 100.0, 150.0, 200.0, 250.0 };
		double gap = gaps[rand() % 5] + double(rand() % 5) * 10.0 - 20.0;
		plateSystem2.changeLampLength(lampL); //default 360 mm
		plateSystem2.generatePlateSystem(sampleNumber); // sampleNumber is a variable within [0, 4999]
		plateSystem2.setLayers(layers, gap, isStaggered);
		if (i == 1)
		{
			plateSystem2.addReceiverY(deltaY);
		}
		LDFField ldfField2(plateSystem2);
		ldfField2.runLDFField(info.str());
#if includeLPF == 1
		ldfField2.runLPFField(info.str());
#endif
		RadField radField2(plateSystem2);
		//radField2.setPhotonNumber(4000);
		radField2.printTotalPhotonNum();
		radField2.runRadiantField(numThreads, info.str());
	}
	

	// ----- 3 ----- 更改recieverY，单层
	std::map<int, int>numLamps3;
	for (int i = 0; i < 2; i++)
	{
		int tmp = 3 + rand() % 10;
		auto t = numLamps3.find(tmp);
		if (t == numLamps3.end()) {
			numLamps3[tmp];
		}
		else {
			i--;
		}
	}
	for (auto it = numLamps3.begin(); it != numLamps3.end(); it++)
	{
		std::stringstream info;
		double deltaY = dy[rand() % 6] + ddy * (double(rand() % 7) - 3.0);
		info << std::setw(3) << std::setfill('0') << int(deltaY);
		PlateSystem plateSystem3(it->first);// Only set the number of lamps. Other parameters are default.
		plateSystem3.generatePlateSystem(sampleNumber); // sampleNumber is a variable within [0, 4999]
		plateSystem3.addReceiverY(deltaY);

		LDFField ldfField3(plateSystem3);
		ldfField3.runLDFField(info.str());
#if includeLPF == 1
		ldfField3.runLPFField(info.str());
#endif

		RadField radField3(plateSystem3);
		//radField3.setPhotonNumber(4000);
		radField3.runRadiantField(numThreads, info.str());
	}
}


#if runMode == 1
void runRadLDFTestMode(unsigned int sampleNumber) {
	int numThreads = 64;
	int numLamps = 3 + rand() % 8;
	PlateSystem plateSystem1(numLamps);// Only set the number of lamps. Other parameters are default.
	plateSystem1.generatePlateSystem(sampleNumber); // sampleNumber is a variable within [0, 4999]

	// Must use the same plateSystem, simultaneously compute
	LDFField ldfField1(plateSystem1);
	ldfField1.runLDFField();

	RadField radField1(plateSystem1);
	//radField1.setPhotonNumber(4000);
	radField1.runRadiantField(numThreads);

	double dy[6] = { 100.0, 180.0, 260.0, 340.0, 420.0, 500.0 };
	double ddy = 10.0;
	// ----- 2 ----- 多层 & 增加Y
	for (int i = 0; i < 3; i++)
	{
		unsigned int layers = i + 1; // 1, 2, 3
		int numLamps2 = layers * 3 + rand() % 10 * layers;
		Sleep(10);
		std::stringstream info;
		double deltaY = dy[rand() % 6] + ddy * (double(rand() % 7) - 3.0);
		info << std::setw(3) << std::setfill('0') << int(deltaY);
		PlateSystem plateSystem2(numLamps2);// Only set the number of lamps. Other parameters are default.
		double lampL = 360.0 / double(layers) + 6.0 * double(rand() % 8);
		double gaps[4] = { 75.0, 150.0, 225.0, 300.0 };
		double gap = gaps[rand() % 4] + double(rand() % 3) * 25.0 - 25.0;
		plateSystem2.changeLampLength(lampL); //default 360 mm
		plateSystem2.generatePlateSystem(sampleNumber); // sampleNumber is a variable within [0, 4999]
		plateSystem2.setLayers(layers, gap, 0);	
		plateSystem2.addReceiverY(deltaY);

		LDFField ldfField2(plateSystem2);
		ldfField2.runLDFField(info.str());
		RadField radField2(plateSystem2);
		//radField2.setPhotonNumber(4000);
		radField2.printTotalPhotonNum();
		radField2.runRadiantField(numThreads, info.str());
	}
}
#endif

#if runMode == 2
void run3DRadLDF(unsigned int sampleNumber) {
	//unsigned int sampleNumber = 0;
	unsigned int numLamps = 6;
	const double dy = 4.0;
	const int zNum = 128;
	double Progress = 0;
	for (unsigned int i = 0; i < zNum; i++)
	{
		std::cout << "\nProgress = " << Progress << "%\n";
		std::stringstream info;
		info << std::setw(3) << std::setfill('0') << int(i * dy);
		PlateSystem plateSystem(numLamps);// Only set the number of lamps. Other parameters are default.
		plateSystem.generatePlateSystem(sampleNumber); // sampleNumber is a variable within [0, 4999]
		plateSystem.addReceiverY(i * dy);

		LDFField ldfField(plateSystem);
		ldfField.runLDFField(info.str());

		RadField radField(plateSystem);
		radField.setPhotonNumber(4000);
		radField.runRadiantField(64, info.str());
		Progress += 1.0 / double(zNum) * 100.0;
	}
}
#endif

#if runMode == 3
void runMultiLayersRadLDF(unsigned int sampleNumber, int numLamps) {
	unsigned int layers = 2;
	bool isStaggered = 1;
	std::stringstream info;
	info << layers << isStaggered;
	PlateSystem plateSystem(numLamps);// Only set the number of lamps. Other parameters are default.
	plateSystem.changeLampLength(150.0); //default 360 mm
	plateSystem.generatePlateSystem(sampleNumber); // sampleNumber is a variable within [0, 4999]
	plateSystem.setLayers(layers, 150.0, isStaggered);
		
	LDFField ldfField(plateSystem);
	ldfField.runLDFField(info.str());

	RadField radField(plateSystem);
	radField.setPhotonNumber(4000);
	radField.runRadiantField(64, info.str());
	
}
#endif

#endif //mainMode