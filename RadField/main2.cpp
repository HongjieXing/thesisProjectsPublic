#include"../RadFieldDLL/pch.h"
#pragma comment(lib, "RadFieldDLL.lib") //需要在:F\RadField\testMain\main.cpp目录拷贝上新生成的lib文件
#if mainMode == 2
/* 
	This Mode can output the heater set ups, including numLamp, sampleNumber, receiverY
 and lampPara for each lamp. These informations are stored in heaterInfo.txt by a single line .

	The info can be read and set the heater by PlateParameter::readAndSetHeater(std::istringstream &)
 line by line for each case.
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

void runRadLDF(unsigned int sampleNumber);
#define includeLPF 1
#define runMode 1	//0:generate heaters; 1:read heaters and run
#define runRad 0	//0:no Rad;  1 run Rad

int main() {

	srand((unsigned int)time(NULL));
	const int MaxSampleNum = 5000;

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
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
//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#if runMode == 0 // generate heaters
	std::ofstream LOG;
	LOG.open("./log/logTrainSampleNumber.txt", std::ios::app | std::ios::out);
	if (!LOG) {
		std::cout << "Error, cannot open logTestSampleNumber file.\n";
	}
	const int sumSample = 10; //test sample number
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

#elif runMode == 1 // read heater and run
	std::ifstream heaterInfo;
	heaterInfo.open("./log/heaterInfo.LOG", std::ios::app | std::ios::out);
	if (!heaterInfo) {
		std::cout << "Error, cannot open heaterInfo.txt file.\n";
		system("pause");
		return 0;
	}
	int sumSample = 0; 
	std::string oneLineHeater;
	std::vector<std::string> heatersInfo;
	std::vector<int> numLampsHeater;
	while (std::getline(heaterInfo, oneLineHeater))
	{
		heatersInfo.emplace_back(oneLineHeater);
		std::istringstream iss(oneLineHeater);
		int tmp = 0;
		iss >> tmp;
		numLampsHeater.emplace_back(tmp);
		sumSample++;
	}
	heaterInfo.close();
	
	assert(sumSample == heatersInfo.size());
	for (int i = 0; i < sumSample; i++)
	{
		std::cout << "\nProgress = " << double(i) / double(sumSample) * 100 << "%\n";
		std::istringstream iss(heatersInfo[i]);
		PlateSystem heater(numLampsHeater[i]);
		heater.readAndSetHeater(iss); //无需生成 generatePlateSystem(X)
		
		// Must use the same plateSystem, simultaneously compute
		LDFField ldfField(heater);
		ldfField.runLDFField();
#if includeLPF == 1
		ldfField.runLPFField();
#endif

#if runRad
		RadField radField(heater);
		radField.runRadiantField(12);
#endif
	}
	
#endif
	system("pause");
	return 0;
}

/* * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * */

#if runMode == 0 // generate heaters
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
	plateSystem1.generatePlateSystem(sampleNumber); // generate 
	plateSystem1.writeHeaterInfo();
	//plateSystem1.readAndSetHeater(iss); //read heater log, no need for PlateSystem::generatePlateSystem()

	// Must use the same plateSystem, simultaneously compute
	LDFField ldfField1(plateSystem1);
	ldfField1.runLDFField();
#if includeLPF == 1
	ldfField1.runLPFField();
#endif

#if runRad
	RadField radField1(plateSystem1);
	//radField1.setPhotonNumber(4000);
	radField1.runRadiantField(numThreads);
	//}
#endif

	double dy[6] = { 80.0, 160.0, 240.0, 320.0, 400.0, 480.0 };
	double ddy = 10.0;
	// ----- 2 ----- 多层
	for (int i = 0; i < 2; i++)
	{
		unsigned int layers = 2 + i;
		int numLamps2 = layers * 3 + rand() % 10 * layers;
		//Sleep(10);
		//bool isStaggered = rand() % 2;
		bool isStaggered = rand() % 2; //设置为不交错
		double deltaY = dy[rand() % 6] + ddy * (double(rand() % 7) - 3.0);
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
		plateSystem2.writeHeaterInfo();
		LDFField ldfField2(plateSystem2);
		ldfField2.runLDFField();
#if includeLPF == 1
		ldfField2.runLPFField();
#endif

#if runRad
		RadField radField2(plateSystem2);
		//radField2.setPhotonNumber(4000);
		radField2.printTotalPhotonNum();
		radField2.runRadiantField(numThreads, info.str());
#endif
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
		double deltaY = dy[rand() % 6] + ddy * (double(rand() % 7) - 3.0);
		PlateSystem plateSystem3(it->first);// Only set the number of lamps. Other parameters are default.
		plateSystem3.generatePlateSystem(sampleNumber); // sampleNumber is a variable within [0, 4999]
		plateSystem3.addReceiverY(deltaY);
		plateSystem3.writeHeaterInfo();

		LDFField ldfField3(plateSystem3);
		ldfField3.runLDFField();
#if includeLPF == 1
		ldfField3.runLPFField();
#endif

#if runRad
		RadField radField3(plateSystem3);
		//radField3.setPhotonNumber(4000);
		radField3.runRadiantField(numThreads, info.str());
#endif
	}
}
#endif


#endif //mainMode