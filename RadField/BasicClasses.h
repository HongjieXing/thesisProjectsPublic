#pragma once
#include"framework.h"

class __declspec(dllexport) DirectVec {
public:
	double a;
	double b;
	double c;
	DirectVec() {
		a = 0.0; b = 0.0; c = 1.0;
	};
	DirectVec(double x, double y, double z) {
		a = x; b = y; c = z;
	};
	DirectVec(const DirectVec& obj) {
		this->a = obj.a;
		this->b = obj.b;
		this->c = obj.c;
	};
	double SecondNorm() const {
		return sqrt(a * a + b * b + c * c);
	};
	void normalization() {
		double secNorm = SecondNorm();
		if (abs(secNorm - 1.0) > 1.0e-9 && secNorm > 1.0e-7) {
			this->a /= secNorm;
			this->b /= secNorm;
			this->c /= secNorm;
		}
	};
	bool isNorm()const {
		if (abs(this->SecondNorm() - 1.0) < 1.0e-9) {
			return 1;
		}
		else
			return 0;
	};
	double dotProduct(const DirectVec& obj) const {
		return this->a * obj.a + this->b * obj.b + this->c * obj.c;
	};
	DirectVec operator *(const double len) {
		DirectVec vec;
		vec.a = len * this->a;
		vec.b = len * this->b;
		vec.c = len * this->c;
		return vec;
	}
	bool isZero() const {
		if (this->SecondNorm() < 1.0e-9) {
			return 1;
		}
		else
			return 0;
	};
	//均按照左手定则旋转
	void rotateXaxis(double alpha) {
		// alpha: rad
		double tmpb = b * cos(alpha) - c * sin(alpha);
		double tmpc = b * sin(alpha) + c * cos(alpha);
		this->b = tmpb;
		this->c = tmpc;
	};
	void rotateYaxis(double beta) {
		// beta: rad
		double tmpa = a * cos(beta) + c * sin(beta);
		double tmpc = -a * sin(beta) + c * cos(beta);
		this->a = tmpa;
		this->c = tmpc;
	};
	void rotateZaxis(double gamma) {
		// theta: rad
		double tmpa = a * cos(gamma) - b * sin(gamma);
		double tmpb = a * sin(gamma) + b * cos(gamma);
		this->a = tmpa;
		this->b = tmpb;
	};	
	void print()const {
		std::cout << "Vec(" << this->a << ", " << this->b << ", " << this->c << ")\n";
	};
};

class __declspec(dllexport) PointCoord {
public:
	double x;
	double y;
	double z;
	PointCoord() {
		x = 0.0; y = 0.0; z = 0.0;
	};
	PointCoord(const PointCoord& obj) {
		this->x = obj.x;
		this->y = obj.y;
		this->z = obj.z;
	};
	PointCoord(double x, double y, double z) {
		this->x = x; this->y = y; this->z = z;
	};
	DirectVec vecToPointEnd(const PointCoord& pointEnd) const{
		DirectVec vec;
		vec.a = pointEnd.x - this->x;
		vec.b = pointEnd.y - this->y;
		vec.c = pointEnd.z - this->z;
		vec.normalization();
		return vec;
	};
	void movePoint(const DirectVec& NormVec, double moveLength) {
		if (NormVec.isZero()) {
			std::cout << "Error, moving direction vector is 0\n";
			system("pause");
			return;
		}
		if (!NormVec.isNorm()) {
			std::cout << "Warning, moving direction vector is not a Norm Vector\n";
		}
		else {
			this->x += moveLength * NormVec.a;
			this->y += moveLength * NormVec.b;
			this->z += moveLength * NormVec.c;
		}
	};
	void print()const {
		std::cout << "Point(" << this->x << ", " << this->y << ", " << this->z << ")\n";
	}
};

