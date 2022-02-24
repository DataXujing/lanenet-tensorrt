// lanenettrt.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
// xj


#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <math.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logging.h"
#include <memory>
#include <numeric> 
#include "postprocess/dbscan.hpp"

//高精度计时
#include "mytimer.h"



using namespace sample;

//
//using DBSCAMSample = DBSCAMSample<float>;
//using Feature = Feature<float>;

//前处理

void preprocess(cv::Mat& img, float data[]) {

	//cv::Mat rgb(img.rows, img.cols, CV_8UC3);
	//cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

	if (img.type() != CV_32FC3) {
		img.convertTo(img, CV_32FC3);
	}
	cv::resize(img, img, cv::Size(512,256), 0, 0, cv::INTER_LINEAR);
	cv::divide(img, cv::Scalar(127.5, 127.5, 127.5), img);
	cv::subtract(img, cv::Scalar(1.0, 1.0, 1.0), img);

	std::vector<float> dstdata;


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			dstdata.push_back(img.at<cv::Vec3f>(i, j)[0]);
			dstdata.push_back(img.at<cv::Vec3f>(i, j)[1]);
			dstdata.push_back(img.at<cv::Vec3f>(i, j)[2]);

		}
	}


	std::copy(dstdata.begin(), dstdata.end(), data);

}


//后处理

/***
 * Gather pixel embedding features via binary segmentation result
 * @param binary_mask
 * @param pixel_embedding
 * @param coords
 * @param embedding_features
 */
void gather_pixel_embedding_features(const cv::Mat &binary_mask, const cv::Mat &pixel_embedding,
	std::vector<cv::Point> &coords,
	std::vector<DBSCAMSample<float>> &embedding_samples) {

	auto image_rows = 256;
	auto image_cols = 512;

	for (auto row = 0; row < image_rows; ++row) {
		auto binary_image_row_data = binary_mask.ptr<uchar>(row);
		auto embedding_image_row_data = pixel_embedding.ptr<cv::Vec4f>(row);
		for (auto col = 0; col < image_cols; ++col) {
			auto binary_image_pix_value = binary_image_row_data[col];
			if (binary_image_pix_value == 255) {
				coords.emplace_back(cv::Point(col, row));
				Feature<float> embedding_features;
				for (auto index = 0; index < 4; ++index) {
					embedding_features.push_back(embedding_image_row_data[col][index]);
				}
				DBSCAMSample<float> sample(embedding_features, CLASSIFY_FLAGS::NOT_CALSSIFIED);
				embedding_samples.push_back(sample);
			}
		}
	}
}


/***
 * simultaneously random shuffle two vector inplace. The two input source vector should have the same size.
 * @tparam T
 * @param src1
 * @param src2
 */
template <typename T1, typename T2>
void simultaneously_random_shuffle(std::vector<T1> src1, std::vector<T2> src2) {

	//CHECK_EQ(src1.size(), src2.size());
	if (src1.empty() || src2.empty()) {
		return;
	}

	// construct index vector of two input src
	std::vector<uint> indexes;
	indexes.reserve(src1.size());
	std::iota(indexes.begin(), indexes.end(), 0);
	std::random_shuffle(indexes.begin(), indexes.end());

	// make copy of two input vector
	std::vector<T1> src1_copy(src1);
	std::vector<T2> src2_copy(src2);

	// random two source input vector via random shuffled index vector
	for (long i = 0; i < indexes.size(); ++i) {
		src1[i] = src1_copy[indexes[i]];
		src2[i] = src2_copy[indexes[i]];
	}
}


/***
 * simultaneously random select part of the two input vector into the two output vector.
 * The two input source vector should have the same size because they have one-to-one mapping
 * relation between the elements in two input vector
 * @tparam T1 : type of input vector src1 which should support default constructor
 *              due to the usage of vector resize function
 * @tparam T2 : type of input vector src2 which should support default constructor
 *              due to the usage of vector resize function
 * @param src1 : input vector src1
 * @param src2 : input vector src2
 * @param select_ratio : select ratio which should within range [0.0, 1.0]
 * @param output1 : selected partial vector of src1
 * @param output2 : selected partial vector of src2
 */
template <typename T1, typename T2>
void simultaneously_random_select(
	const std::vector<T1> &src1, const std::vector<T2> &src2, float select_ratio,
	std::vector<T1>& output1, std::vector<T2>& output2) {

	// check if select ratio is right
	if (select_ratio < 0.0 || select_ratio > 1.0) {
		std::cout << "Select ratio should be in range [0.0, 1.0]";
		return;
	}

	// calculate selected element counts using ceil to get
	//CHECK_EQ(src1.size(), src2.size());
	auto src_element_counts = src1.size();
	auto selected_elements_counts = static_cast<uint>(std::ceil(src_element_counts * select_ratio));
	//CHECK_LE(selected_elements_counts, src_element_counts);

	// random shuffle indexes
	std::vector<uint> indexes = std::vector<uint>(src_element_counts);
	std::iota(indexes.begin(), indexes.end(), 0);
	std::random_shuffle(indexes.begin(), indexes.end());

	// select part of the elements via first selected_elements_counts index in random shuffled indexes vector
	output1.resize(selected_elements_counts);
	output2.resize(selected_elements_counts);

	for (uint i = 0; i < selected_elements_counts; ++i) {
		output1[i] = src1[indexes[i]];
		output2[i] = src2[indexes[i]];
	}
}


/***
 * Calculate the mean feature vector among a vector of DBSCAMSample samples
 * @param input_samples
 * @return
 */
Feature<float> calculate_mean_feature_vector(const std::vector<DBSCAMSample<float>> &input_samples) {

	if (input_samples.empty()) {
		return Feature<float>();
	}

	uint feature_dims = input_samples[0].get_feature_vector().size();
	uint sample_nums = input_samples.size();
	Feature<float> mean_feature_vec;
	mean_feature_vec.resize(feature_dims, 0.0);
	for (const auto& sample : input_samples) {
		for (uint index = 0; index < feature_dims; ++index) {
			mean_feature_vec[index] += sample[index];
		}
	}
	for (uint index = 0; index < feature_dims; ++index) {
		mean_feature_vec[index] /= sample_nums;
	}

	return mean_feature_vec;
}

/***
 *
 * @param input_samples
 * @param mean_feature_vec
 * @return
 */
Feature<float> calculate_stddev_feature_vector(
	const std::vector<DBSCAMSample<float>> &input_samples,
	const Feature<float>& mean_feature_vec) {

	if (input_samples.empty()) {
		return Feature<float>();
	}

	uint feature_dims = input_samples[0].get_feature_vector().size();
	uint sample_nums = input_samples.size();

	// calculate stddev feature vector
	Feature<float> stddev_feature_vec;
	stddev_feature_vec.resize(feature_dims, 0.0);
	for (const auto& sample : input_samples) {
		for (uint index = 0; index < feature_dims; ++index) {
			auto sample_feature = sample.get_feature_vector();
			auto diff = sample_feature[index] - mean_feature_vec[index];
			diff = std::pow(diff, 2);
			stddev_feature_vec[index] += diff;
		}
	}
	for (uint index = 0; index < feature_dims; ++index) {
		stddev_feature_vec[index] /= sample_nums;
		stddev_feature_vec[index] = std::sqrt(stddev_feature_vec[index]);
	}

	return stddev_feature_vec;
}


/***
 * Normalize input samples' feature. Each sample's feature is normalized via function as follows:
 * feature[i] = (feature[i] - mean_feature_vector[i]) / stddev_feature_vector[i].
 * @param input_samples
 * @param output_samples
 */
void normalize_sample_features(const std::vector<DBSCAMSample<float>> &input_samples,
	std::vector<DBSCAMSample<float>> &output_samples) {
	// calcualte mean feature vector
	Feature<float> mean_feature_vector = calculate_mean_feature_vector(input_samples);

	// calculate stddev feature vector
	Feature<float> stddev_feature_vector = calculate_stddev_feature_vector(input_samples, mean_feature_vector);

	std::vector<DBSCAMSample<float>> input_samples_copy = input_samples;
	for (auto& sample : input_samples_copy) {
		auto feature = sample.get_feature_vector();
		for (long index = 0; index < feature.size(); ++index) {
			feature[index] = (feature[index] - mean_feature_vector[index]) / stddev_feature_vector[index];
		}
		sample.set_feature_vector(feature);
	}
	output_samples = input_samples_copy;
}


/***
 *
 * @param embedding_samples
 * @param cluster_ret
 */
void cluster_pixem_embedding_features(std::vector<DBSCAMSample<float>> &embedding_samples,
	std::vector<std::vector<uint> > &cluster_ret, std::vector<uint>& noise) {

	if (embedding_samples.empty()) {
		std::cout << "Pixel embedding samples empty";
		return;
	}

	// dbscan cluster
	auto dbscan = DBSCAN<DBSCAMSample<float>, float>();
	uint lanenet_pix_embedding_feature_dims = 4;
	float dbscan_eps = 0.4;  //0.4
	uint dbscan_min_pts = 500;  //500
	std::cout << "耗时的计算开始了！！！！" << std::endl;
	MyTimer mt;
	mt.Start();
	dbscan.Run(&embedding_samples, lanenet_pix_embedding_feature_dims, dbscan_eps, dbscan_min_pts);  // 基于密度的聚类的超参数
	cluster_ret = dbscan.Clusters;
	noise = dbscan.Noise;
	mt.End();
	std::cout << "dbscan聚类耗时：" << mt.costTime / 1000 << "ms" << std::endl;
}

/***
 * Visualize instance segmentation result
 * @param cluster_ret
 * @param coords
 */
void visualize_instance_segmentation_result(
	const std::vector<std::vector<uint> > &cluster_ret,
	const std::vector<cv::Point> &coords,
	cv::Mat& intance_segmentation_result) {

	std::map<int, cv::Scalar> color_map = {
		{0, cv::Scalar(0, 0, 255)},
		{1, cv::Scalar(0, 255, 0)},
		{2, cv::Scalar(255, 0, 0)},
		{3, cv::Scalar(255, 0, 255)},
		{4, cv::Scalar(0, 255, 255)},
		{5, cv::Scalar(255, 255, 0)},
		{6, cv::Scalar(125, 0, 125)},
		{7, cv::Scalar(0, 125, 125)}
	};

	for (long class_id = 0; class_id < cluster_ret.size(); ++class_id) {
		auto class_color = color_map[class_id];
#pragma omp parallel for
		for (auto index = 0; index < cluster_ret[class_id].size(); ++index) {
			auto coord = coords[cluster_ret[class_id][index]];
			auto image_col_data = intance_segmentation_result.ptr<cv::Vec3b>(coord.y);
			image_col_data[coord.x][0] = class_color[0];
			image_col_data[coord.x][1] = class_color[1];
			image_col_data[coord.x][2] = class_color[2];
		}
	}
}


// detect
void detect( cv::Mat &binary_seg_result, cv::Mat &instance_seg_result,float* h_output_pix, float* h_output_binary) {

	//int sz[3] = { 512,256,4 };
	//cv::Mat pix_embedding_output_mat(3, sz, CV_32FC4, cv::Scalar::all(0));
	cv::Mat pix_embedding_output_mat(cv::Size(512,256), CV_32FC4);
	//memcpy(pix_embedding_output_mat.data, h_output_pix, 256 * 512 * 4);
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 512; j++) {
			for (int k = 0; k < 4; k++) {
				pix_embedding_output_mat.at<cv::Vec4f>(i, j)[k] = h_output_pix[i * 512 * 4 + j * 4 + k];
			}
		}
	}


	cv::Mat binary_output_mat = cv::Mat_<float>(256, 512);
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 512; j++) {
			binary_output_mat.at<float>(i, j) = h_output_binary[i * 512 + j] * 255;
		}
	}

	binary_output_mat.convertTo(binary_seg_result, CV_8UC1);


	// gather pixel embedding features
	std::vector<cv::Point> coords;
	std::vector<DBSCAMSample<float>> pixel_embedding_samples;
	gather_pixel_embedding_features(binary_seg_result, pix_embedding_output_mat, coords, pixel_embedding_samples);

	// simultaneously random shuffle embedding vector and coord vector inplace
	simultaneously_random_shuffle<cv::Point, DBSCAMSample<float>>(coords, pixel_embedding_samples);

	// simultaneously random select embedding vector and coord vector to reduce the cluster time
	std::vector<cv::Point> coords_selected;
	std::vector<DBSCAMSample<float>> pixel_embedding_samples_selected;
	simultaneously_random_select<DBSCAMSample<float>, cv::Point>(pixel_embedding_samples, coords,
		0.5, pixel_embedding_samples_selected, coords_selected); //0.5:embedding特征抽稀比例
	coords.clear();
	coords.shrink_to_fit();
	pixel_embedding_samples.clear();
	pixel_embedding_samples.shrink_to_fit();

	// normalize pixel embedding features
	normalize_sample_features(pixel_embedding_samples_selected, pixel_embedding_samples_selected);

	// cluster samples
	std::vector<std::vector<uint> > cluster_ret;
	std::vector<uint> noise;
	cluster_pixem_embedding_features(pixel_embedding_samples_selected, cluster_ret, noise);


	// visualize instance segmentation
	instance_seg_result = cv::Mat(cv::Size(512,256), CV_8UC3, cv::Scalar(0, 0, 0));
	visualize_instance_segmentation_result(cluster_ret, coords_selected, instance_seg_result);
}


float h_input_0[256 * 512 * 3]; //lanenet/input_tensor:0
float h_output_binary[256 * 512];   //lanenet/final_binary_output:0
float h_output_pix[256 * 512 * 4];   //lanenet/final_pixel_embedding_output:0


int main()
{
	Logger gLogger;
	////初始化插件，调用plugin必须初始化plugin respo
	//nvinfer1:initLibNvInferPlugins(&gLogger, "");


	nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(gLogger);
	std::string engine_filepath = "./model/lanenet.engine";

	std::ifstream file;
	file.open(engine_filepath, std::ios::binary | std::ios::in);
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	file.seekg(0, std::ios::beg);

	std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
	file.read(data.get(), length);
	file.close();

	//nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
	nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(data.get(), length, nullptr);
	nvinfer1::IExecutionContext* engine_context = engine_infer->createExecutionContext();

	int input_index = engine_infer->getBindingIndex("lanenet/input_tensor:0"); //1x256*512*3
	int output_index_1 = engine_infer->getBindingIndex("lanenet/final_binary_output:0");  //2
	int output_index_2 = engine_infer->getBindingIndex("lanenet/final_pixel_embedding_output:0");   // 1



	std::cout << "输入的index: " << input_index << " 输出的lanenet/final_binary_output:0-> " << output_index_1 << 
		" 输出的lanenet/final_pixel_embedding_output:0-> " << output_index_2 << std::endl;

	if (engine_context == nullptr)
	{
		std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
	}

	// cached_engine->destroy();
	std::cout << "loaded trt model , do inference" << std::endl;


	////cv2读图片
	//cv::Mat image;
	//image = cv::imread("./test.jpg", 1);
	//int w = image.cols;
	//int h = image.rows;

	// 读取文件夹中的图片
	cv::String pattern = "test/*.*";
	std::vector<cv::String> fn;
	cv::glob(pattern, fn, false);
	size_t count = fn.size();  //图片个数


	for (size_t i = 0; i < count; i++)
	{

		std::cout << "-------------------------------" << std::endl;
		std::cout << "图片：" << fn[i] << std::endl;
		cv::Mat image = cv::imread(fn[i]);
		cv::Mat source_img = image.clone();

		preprocess(image, h_input_0);

		void* buffers[3];
		cudaMalloc(&buffers[0], 256 * 512 * 3 * sizeof(float));  //<- lanenet/input_tensor:0

		cudaMalloc(&buffers[1], 256 * 512 * 4 * sizeof(float)); //<- lanenet/final_pixel_embedding_output:0
		cudaMalloc(&buffers[2], 256 * 512 * sizeof(float)); //<- lanenet/final_binary_output:0


		cudaMemcpy(buffers[0], h_input_0, 256 * 512 * 3 * sizeof(float), cudaMemcpyHostToDevice);

		MyTimer mt;
		mt.Start();
		// ----------- do execute --------//
		engine_context->executeV2(buffers);

		cudaMemcpy(h_output_pix, buffers[1], 256 * 512 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output_binary, buffers[2], 256 * 512 * sizeof(float), cudaMemcpyDeviceToHost);
		mt.End();
		std::cout << "推断时间： " << mt.costTime / 1000 << "ms" << std::endl;

		//-------------------------------------后处理--------------------------------------------

		cv::Mat binary_mask;
		cv::Mat instance_mask;

		detect(binary_mask, instance_mask, h_output_pix, h_output_binary);

		cv::imwrite("binary_ret.png", binary_mask);
		cv::imwrite("instance_ret.png", instance_mask);

		// 图像融合
		int w = source_img.cols;
		int h = source_img.rows;
		cv::Mat instance_mask_resize;
		cv::resize(instance_mask, instance_mask_resize, cv::Size(w,h));


		cv::addWeighted(source_img, 0.6, instance_mask_resize, 0.4, 0, source_img);//0.5+0.5=1,0.3+0.7=1

		cv::imwrite("res.jpg", source_img);



		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
		cudaFree(buffers[2]);

	}

	engine_runtime->destroy();
	engine_infer->destroy();

	return 0;
}




