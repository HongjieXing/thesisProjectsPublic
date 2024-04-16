#pragma once
#include"framework.h"
#include"BasicClasses.h"
const double PI = 3.14159;

#define LPFMode 2 //1 or 2; else error

class __declspec(dllexport) Lamp
{
	double lampR;
	double lampL;
	PointCoord centerCoord;
	DirectVec lampVec;
	double angle_z2x;
	double angle_z2y;

	double emitRangeMin;
	double emitRangeMax;
public:
	Lamp();
	Lamp(const Lamp& lamp);
	//Lamp& operator=(const Lamp& lamp);
	const double& getLampR() const { return this->lampR; };
	const double& getLampL() const { return this->lampL; };
	const PointCoord& getCenterCoord() const { return this->centerCoord; };
	const DirectVec& getLampVec() const { return this->lampVec; };
	const double& getAngle_z2x() const { return this->angle_z2x; };
	const double& getAngle_z2y() const { return this->angle_z2y; };
	const double& getEmitRangeMin() const { return this->emitRangeMin; };
	const double& getEmitRangeMax() const { return this->emitRangeMax; };
	const PointCoord getLampEnd(const int& whichEnd)const; //whichEnd = 0 or !0
	double YDisInLampRange(const PointCoord& gridCoord)const;//return y dis if in lamp range

	void rotateAngle_z2x(const unsigned int& randInt_0to2);
	void rotateAngle_z2y(const unsigned int& randInt_0to3);
	void changeLampLength(const double& lampLength);
	void changeLampPara(const unsigned int& randInt_0to9);
	void calcuLampVec();
	void calcuLampVec(DirectVec& directVec);
	void setCentralCoord(const PointCoord& centralPoint);
	void setCentralCoordZ(const double& Z);
	void setCentralCoordX(const double& X);
	void writeLampInfo(std::ofstream& LOG)const;
	void printLampInfo()const;
	bool readAndSetLamp(std::istringstream& lampInfo);
	/*
	* Definate in class "PlateSystem"
	* void changeCentralCoord(const unsigned int& randInt_0to9);
	*/
};

struct __declspec(dllexport) PlateParameter {
	double Length_x;	// >0 half length
	double Distance_y;
	double Length_z;	// >0 half length
	PlateParameter() {
		this->Distance_y = 0.0;
		this->Length_x = 0.0;
		this->Length_z = 0.0;
	}
	PlateParameter(const double& x, const double& y, const double& z) {
		this->Distance_y = y;
		this->Length_x = x;
		this->Length_z = z;
	}
	PlateParameter(const PlateParameter& obj) {
		this->Distance_y = obj.Distance_y;
		this->Length_x = obj.Length_x;
		this->Length_z = obj.Length_z;
	}
};

class __declspec(dllexport) PlateSystem
{
	const unsigned int numLamp;
	Lamp* lampArray;
	PlateParameter Receiver;
	PlateParameter Reflector;
	double reflectivity;
	unsigned int sampleNumber;

public:
	PlateSystem(unsigned int numLamps);
	PlateSystem(const PlateSystem& heater);
	PlateSystem(unsigned int numLamps, const PlateParameter& receiver, 
		const PlateParameter& reflector, Lamp* lamparray);
	~PlateSystem();

	void setReflectivity(double reflectivity);
	void setSampleNumber(unsigned int rand_0to4999);
	const double& getReflectivity() const;
	const Lamp& getLampPara(unsigned int index) const;
	/* LOG∏Ò Ω:
	numLamp, sampleNumber, Receiver.Distance_y	(3)
	lampL, x, y, z, angle_z2x, angle_z2y		(6*numLamp)
	(one line for one heater)
	*/
	void writeHeaterInfo();
	void printHeaterInfo();
	void readAndSetHeater(std::istringstream &oneLineHeater);
	const PlateParameter& getReceiver() const;
	const PlateParameter& getReflector() const;
	const unsigned int getNumLamps() const;
	const unsigned int getsampleNumber() const;
	void sampleNumberPlusOne();

	//Operate the Lamp* lampArray
	void rotateLamps(unsigned int z2x_0to2, unsigned int z2y_0to3);
	void rotateLamps(unsigned int randInt_0to9);
	void changeLampLength(const double& lampL);
	void changeLampPara(unsigned int randInt_0to9);
	void changeLampsCentralCoord(const unsigned int randInt_0to9);
	void generateLampArray(unsigned int int1_09, unsigned int int2_09);
	void changeReceiverCoord(const unsigned int randInt_0to4);
	void addReceiverY(double deltaY);
	void generatePlateSystem();
	void generatePlateSystem(unsigned int rand_0to4999);

	void setLayers(unsigned int layer, double gap, bool isStaggered);
};