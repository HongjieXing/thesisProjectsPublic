#pragma once
#include"framework.h"
#include"Heater.h"

#define reflectMode 1	// 1:镜面反射， 2:漫反射
class __declspec(dllexport) Photon
{
	double theta;
	double omega, phi;
	PointCoord emitP;//emitting point
	DirectVec emitV;//emitting direction vector
	PointCoord interP;//intersection direction vector
	DirectVec interV;//intersection point
	PointCoord reflectP;//光子与反射屏的交点
	DirectVec reflectV;//被反射光子的发射方向
	const PlateSystem& plateSystem;
	const Lamp& lamp;

	double* randArray0_1;

	bool isExist;
	bool isReflected;
	bool reachReceiver;
public:
	Photon(const PlateSystem& plateSystem, double* R, unsigned int numoflamp);
	~Photon();
	double randomNum() const;
	double randomNum(int i) const;
	//void setRandomArr();
	double randomNum(double min, double max) const;
	void set_theta();
	void set_theta(double min, double max);
	void cal_emitting_point();
	void cal_emitting_angle();
	void cal_emitting_direction();
	inline bool Exist() const { return this->isExist; };
	inline bool Reflected() const { return this->isReflected; };
	inline bool ReachReceiver() const { return this->reachReceiver; };

	// 圆柱
	double is_intersect();
	void cal_intersection_point_cyl(double t);
	void cal_intersection_direction(double D, double Rengine);

	// 平板
	void cal_reflection_direction();
	//void cal_intersectionP_plate(double t);
	void filament_to_reflector();
	bool canbeReflected();
	bool is_blocked();
	void filament_to_receiver();
	void reflector_to_receiver();
	//void cal_intersectionP_receiver(double t);

	void Run_PlateCase();
	void read_inter_p(PointCoord& intersectPoint)const;
	void read_inter_p(double& x, double& y, double& z)const;
	void read_inter_p(double& x, double& z)const;
};