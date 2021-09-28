/**
 * @file parallelproj_cuda.h
 */

#ifndef __PARALLELPROJ_CUDA_H__
#define __PARALLELPROJ_CUDA_H__

/** @brief 3D non-tof joseph forward projector CUDA wrapper
 *
 *  @param h_xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param h_xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param d_img    Pointer to device arrays of shape [n0*n1*n2] containing the 3D image to 
 *                  be projected.
 *                  The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *  @param h_img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param h_voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param h_p           array of length nlors (output) used to store the projections
 *  @param nlors           number of projections (length of p array)
 *  @param h_img_dim       array with dimensions of image [n0,n1,n2]
 *  @param threadsperblock number of threads per block
 */
extern "C" void joseph3d_fwd_cuda(const float *h_xstart, 
                                  const float *h_xend, 
                                  float **d_img,
                                  const float *h_img_origin, 
                                  const float *h_voxsize, 
                                  float *h_p,
                                  long long nlors, 
                                  const int *h_img_dim,
                                  int threadsperblock);

#endif
