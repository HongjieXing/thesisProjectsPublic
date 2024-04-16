#pragma once
#include"framework.h"
#include"Photon.h"
#include<vector>
class __declspec(dllexport) Field {
protected:
	double* data;
	const PlateSystem& plateSystem;
	std::ofstream RESULT;
	std::ofstream LOG;
private:
	bool saveResultTmp(std::string& filePath, bool isBinary);
public:
	Field(const PlateSystem& plateSystem);
	virtual ~Field();	
	bool saveResult(std::string& filePath, bool isBinary);
	bool saveResult(std::string& filePath, bool isBinary, const std::string& info);
	inline const double& getData(unsigned int index_i, unsigned int index_j) const;
	inline const static int getRows() { return ROWS; }
	inline const static int getCols() { return COLS; }
	inline const PlateSystem* getPlateSystem()const;
	void writeLog();
	virtual void writeLog(std::string info);
	std::string timeNow()const;
};


class __declspec(dllexport) LDFField : public Field
{
	std::string pathLDF;
	void saveLDFResult(bool isBinary);
	void saveLPFResult(bool isBinary);
	void saveLDFResult(bool isBinary, const std::string& info);
	void saveLPFResult(bool isBinary, const std::string& info);
public:
	LDFField(const PlateSystem& plateSystem);
	~LDFField() { };
	/*				 ~~				 ~~
				  (.)  ~~		   (`) ~~
				 (  .)  ~~		  (.  ) ~~
				(. .  )  ~~		 ( ^  .) ~~
			   (_______)  ~~	(_______) ~~
		shit mountain
		taste and paste(CV) more shit
	*/
	double calcuGrid2LampAngle(const PointCoord& centralGrid, const Lamp& lamp);
	double calcuPoint2LampDistance(const PointCoord& centralGrid, const Lamp& lamp);
	double calcuYDisInLampRangeForLPF(const PointCoord& centralGrid);
	void calcuLPFField();
	void calcuLDFField();
	//void normalizationLDF();	
	// calculate and copy to Pointer LDF or LPF
	bool runLDFField(double* const LDF);
	bool runLPFField(double* const LPF);
	// calculate and save to file
	void runLDFField();
	void runLDFField(const std::string& info);
	void runLPFField();
	void runLPFField(const std::string& info);
	void writeLog(double diffTime);
	void writeLog(std::string info);
};

class __declspec(dllexport) RadField : public Field
{
	unsigned long int numGridAvePhoton;
	std::string pathRad;
	size_t sumP;
	size_t totalPhotonNum;
	void saveResult(bool isBinary);
	void saveResult(bool isBinary, const std::string& info);
public:
	RadField(const PlateSystem& plateSystem);
	~RadField() { };
	void setPhotonNumber(unsigned int aveGridPhotonNumber) {
		this->numGridAvePhoton = aveGridPhotonNumber;
	};	
	Photon generatePhoton(unsigned int numoflamp);
	void calcuRadiantField(unsigned int numThreads);
	void runRadiantField(unsigned int numThreads);
	void runRadiantField(unsigned int numThreads, const std::string& info);
	double gridHeatFlux(double num, double dx, double dz, double photonPower);
	void writeLog(double diffTime);
	void writeLog(std::string info);
	void printTotalPhotonNum()
	{
		std::cout << "TotalPhotonNum = " << totalPhotonNum << std::endl;
	}
};