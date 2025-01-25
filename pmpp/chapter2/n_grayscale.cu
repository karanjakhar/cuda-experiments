#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <png.h>
#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}


__global__ void grayscaleKernel(unsigned char * input, unsigned char * output, int rows, int cols){

    int tid = threadIdx.x  + blockDim.x * blockIdx.x;
    tid = tid * 4;

    int out_tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(out_tid < (rows * cols)){

    output[out_tid] = (unsigned char)(input[tid] * 0.21f + input[tid+1] * 0.72f + input[tid+2] * 0.07f);
    }
    

}



// Function to read a PNG file using libpng
int loadPNG(const char* filename, unsigned char** image, int* width, int* height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return 0;
    }

    png_byte header[8];
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8)) {
        fprintf(stderr, "Error: File is not a valid PNG.\n");
        fclose(fp);
        return 0;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Error: png_create_read_struct failed.\n");
        fclose(fp);
        return 0;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Error: png_create_info_struct failed.\n");
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        return 0;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error: libpng encountered an error.\n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return 0;
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) {
        png_set_strip_16(png);
    }
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }
    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);

    int rowbytes = png_get_rowbytes(png, info);
    *image = (unsigned char*)malloc(rowbytes * (*height));
    if (!*image) {
        fprintf(stderr, "Error: Failed to allocate memory for image.\n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return 0;
    }

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    for (int y = 0; y < *height; y++) {
        row_pointers[y] = *image + y * rowbytes;
    }
    png_read_image(png, row_pointers);

    free(row_pointers);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return 1;
}

// Function to save a grayscale image to a PNG file
int savePNG(const char* filename, const unsigned char* grayscale, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
        return 0;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Error: png_create_write_struct failed.\n");
        fclose(fp);
        return 0;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Error: png_create_info_struct failed.\n");
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return 0;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error: libpng encountered an error during writing.\n");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return 0;
    }

    png_init_io(png, fp);

    // Set PNG header
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // Write image data
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)(grayscale + y * width);
    }
    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    free(row_pointers);
    png_destroy_write_struct(&png, &info);
    fclose(fp);

    return 1;
}






int main(){


    const char* filename = "input.png";
    const char* output_filename = "grayscale_image.png";
    unsigned char* image = NULL;
    int width, height;

    // Step 1: Load the PNG file
    if (!loadPNG(filename, &image, &width, &height)) {
        return -1;
    }

    printf("Loaded PNG: %s, Width: %d, Height: %d\n", filename, width, height);

    int rows = height;
    int cols = width;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;


    size_t matrixSize = rows * cols * 4 * sizeof(unsigned char);
    size_t outputMatrixSize = rows * cols * sizeof(unsigned char); 

    // unsigned char *input = (unsigned char *)malloc(matrixSize);

    unsigned char *output = (unsigned char *)malloc(outputMatrixSize);

    // for(int i = 0; i < matrixSize; i++){
    //     input[i] = (unsigned char) (rand() % 256);
    // }

    unsigned char *input_d, *output_d;

    cudaEventRecord(start);
    cudaMalloc(&input_d, matrixSize);
    cudaMalloc(&output_d, outputMatrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("GPU memory allocation time: %f ms\n", ms);

    cudaEventRecord(start);

    CUDA_CHECK(cudaMemcpy(input_d, image, matrixSize, cudaMemcpyHostToDevice));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    printf("Host to device memory transfer time: %f ms\n", ms);

    int blocks = rows;
    int threads = cols;

    cudaEventRecord(start);
    grayscaleKernel<<<blocks, threads>>>(input_d, output_d, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel excution time: %f ms\n", ms);

    

    // cudaDeviceSynchronize();

    cudaEventRecord(start);
    cudaMemcpy(output, output_d, outputMatrixSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Device to host memory transfer time: %f ms\n", ms);

    if (savePNG(output_filename, output, width, height)) {
        printf("Grayscale image saved to: %s\n", output_filename);
    } else {
        fprintf(stderr, "Error saving grayscale image.\n");
    }

    cudaFree(input_d);
    cudaFree(output_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(image);
    free(output);


    return 0;


}



