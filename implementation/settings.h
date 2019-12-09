#ifndef SETTINGSH
#define SETTINGSH

#include "ray.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cout << "CUDA error" << std::endl;
		cudaDeviceReset();
		exit(99);
	}
}

__device__ vec3 random_direction(curandState *local_rand_state) {
	vec3 p;
	do {
		p = 2.0*vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))
			- vec3(1, 1, 1);
	} while (p.squared_length() >= 2.0);
	return p;
}

class camera {
public:
	__device__ camera() {
		lower_left_corner = vec3(-2.0, -1.0, -1.0);
		horizontal = vec3(4.0, 0.0, 0.0);
		vertical = vec3(0.0, 2.0, 0.0);
		origin = vec3(1, 0.0, 2.0);
	}
	__device__ ray get_ray(float u, float v) { return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin); }

	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
};

#endif