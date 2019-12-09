#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "hitable_list.h"
#include "settings.h"

__device__ vec3 color(ray& r, hitable_list **world, curandState *local_rand_state) {
	ray cur_ray = r;
	float cur_attenuation = 1.0;
	vec3 tmp = random_direction(local_rand_state);
	for (int i = 0; i < 5; i++) { // this is to find the degree of darkening
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.0001, FLT_MAX, rec)) {
			cur_attenuation *= 0.5;
			cur_ray = ray(rec.p, rec.normal + tmp);
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5 * (unit_direction.x() + 1.0);
			vec3 c = (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); 
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curand_init(pixel_index, 0 , 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int k, camera **cam, hitable_list **world, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) 
		return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r = (*cam)->get_ray(u, v);
	for (int t = 0; t < k; t++) //brute force k random directions to find the degree of darkening
		col += color(r, world, &local_rand_state);
	rand_state[pixel_index] = local_rand_state;
	col /= float(k);
	fb[pixel_index] = col;
}

__global__ void create_world(hitable **d_list, hitable_list **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_list[0] = new sphere(vec3(0, 0, -1), 0.5);
		d_list[1] = new sphere(vec3(0, -100.5, -1), 100);
		d_list[2] = new sphere(vec3(1, 0, -1), 0.5);
		*d_world = new hitable_list(d_list, 3);
		*d_camera = new camera();
	}
}

__global__ void free_world(hitable **d_list, hitable_list **d_world, camera **d_camera) {
	delete d_list[0];
	delete d_list[1];
	delete d_list[2];
	delete *d_world;
	delete *d_camera;
}

int main() {
	int x = 600;
	int y = 300;
	int k = 12;
	int tx = 8;
	int ty = 8;

	std::cout << "Rendering a " << x << "x" << y << " image with " << k << " samples per pixel " << std::endl;

	int num_pixels = x * y;
	size_t fb_size = num_pixels * sizeof(vec3);


	vec3 *fb;
	hitable_list **d_world;
	camera **d_camera;
	hitable **d_list;
	curandState *d_rand_state;
	checkCudaErrors(cudaMallocManaged(&fb, fb_size));
	checkCudaErrors(cudaMalloc(&d_world, sizeof(hitable_list *)));
	checkCudaErrors(cudaMalloc(&d_camera, sizeof(camera *)));
	checkCudaErrors(cudaMalloc(&d_list, 3 * sizeof(hitable *)));
	checkCudaErrors(cudaMalloc(&d_rand_state, num_pixels * sizeof(curandState)));
	create_world <<< 1, 1 >>> (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();
	dim3 blocks(x / tx + 1, y / ty + 1);
	dim3 threads(tx, ty);
	render_init <<< blocks, threads >>> (x, y, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render <<< blocks, threads >>> (fb, x, y, k, d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "took " << timer_seconds << " seconds" << std::endl;

	FILE * f = fopen("cuda.ppm", "w");
	fprintf(f, "P3\n%d %d\n%d\n", x, y, 255);
	for (int j = y - 1; j >= 0; j--) {
		for (int i = 0; i < x; i++) {
			size_t pixel_index = j * x + i;
			int ir = int(255.99*fb[pixel_index].r());
			int ig = int(255.99*fb[pixel_index].g());
			int ib = int(255.99*fb[pixel_index].b());
			fprintf(f, "%d %d %d ", ir, ig, ib);
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());
	free_world <<< 1, 1 >>> (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
}