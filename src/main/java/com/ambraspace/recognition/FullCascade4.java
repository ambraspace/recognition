package com.ambraspace.recognition;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RGBA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_videoio.CAP_V4L2;
import static org.bytedeco.javacpp.opencv_highgui.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.bytedeco.javacpp.opencv_core.Point;


public class FullCascade4 {
	
	private static long numOfFrames = 0;
	
	private static final Logger log = LoggerFactory.getLogger(FullCascade4.class);
	
	private final MultiLayerNetwork net13detection;
	private final MultiLayerNetwork net25detection;
	private final MultiLayerNetwork net51detection;
	private final MultiLayerNetwork net13calibration;
	private final MultiLayerNetwork net25calibration;
	private final MultiLayerNetwork net51calibration;
	
	private final MultiLayerNetwork classification;
	
	
	double thresholdT1 = 0.001;
	double thresholdT2 = 0.018;
	double thresholdT3 = 0.3;
	double calibrationThreshold = 0.2;
	double classificationThreshold = 0.5; //0.9653;
	double minIoU = 0.5;

	
	ImagePreProcessingScaler ipps;
	
	
	private final int stage1KernelSize = 13;
	private final int stage1Stride = 3;

	private final int stage2KernelSize = 25;

	private final int stage3KernelSize = 51;
	
	private final double zoomFactor = 1.46;
	
	int batchSize = 100;

	private static double[][] calibrationParams = {
			{-2, -1.0/6, -1.0/6 },
			{-2, -1.0/6,      0 },
			{-2, -1.0/6,  1.0/6 },
			{-2,      0, -1.0/6 },
			{-2,      0,      0 },
			{-2,      0,  1.0/6 },
			{-2,  1.0/6, -1.0/6 },
			{-2,  1.0/6,      0 },
			{-2,  1.0/6,  1.0/6 },
			{-1, -1.0/6, -1.0/6 },
			{-1, -1.0/6,      0 },
			{-1, -1.0/6,  1.0/6 },
			{-1,      0, -1.0/6 },
			{-1,      0,      0 },
			{-1,      0,  1.0/6 },
			{-1,  1.0/6, -1.0/6 },
			{-1,  1.0/6,      0 },
			{-1,  1.0/6,  1.0/6 },
			{ 1, -1.0/6, -1.0/6 },
			{ 0, -1.0/6,      0 },
			{ 0, -1.0/6,  1.0/6 },
			{ 0,      0, -1.0/6 },
			{ 0,      0,      0 },
			{ 0,      0,  1.0/6 },
			{ 0,  1.0/6, -1.0/6 },
			{ 0,  1.0/6,      0 },
			{ 0,  1.0/6,  1.0/6 },
			{ 1, -1.0/6, -1.0/6 },
			{ 1, -1.0/6,      0 },
			{ 1, -1.0/6,  1.0/6 },
			{ 1,      0, -1.0/6 },
			{ 1,      0,      0 },
			{ 1,      0,  1.0/6 },
			{ 1,  1.0/6, -1.0/6 },
			{ 1,  1.0/6,      0 },
			{ 1,  1.0/6,  1.0/6 },
			{ 2, -1.0/6, -1.0/6 },
			{ 2, -1.0/6,      0 },
			{ 2, -1.0/6,  1.0/6 },
			{ 2,      0, -1.0/6 },
			{ 2,      0,      0 },
			{ 2,      0,  1.0/6 },
			{ 2,  1.0/6, -1.0/6 },
			{ 2,  1.0/6,      0 },
			{ 2,  1.0/6,  1.0/6 }
	};


	
	
	public FullCascade4(
			MultiLayerNetwork net13detection,
			MultiLayerNetwork net13calibration,
			MultiLayerNetwork net25detection,
			MultiLayerNetwork net25calibration,
			MultiLayerNetwork net51detection,
			MultiLayerNetwork net51calibration,
			MultiLayerNetwork classification) {
		
		this.net13detection = net13detection;
		this.net25detection = net25detection;
		this.net51detection = net51detection;
		this.net13calibration = net13calibration;
		this.net25calibration = net25calibration;
		this.net51calibration = net51calibration;
		this.classification = classification;
		
		ipps = new ImagePreProcessingScaler();
		
		for (int i=0; i<calibrationParams.length; i++) {
			calibrationParams[i][1]=calibrationParams[i][1]+(1-Math.pow(1.1, calibrationParams[i][0]))/2.0;
			calibrationParams[i][2]=calibrationParams[i][2]+(1-Math.pow(1.1, calibrationParams[i][0]))/2.0;
		}
		
	}
	

	public List<SortedFrame> detectFaces(Mat image, int minFaceSize, int maxFaceSize) throws IOException {
		
		Mat convertedImage = new Mat();
		
		long start, stop;
		
        if (image.channels() != 1) {
            int code = -1;
            switch (image.channels()) {
                case 3:
                    code = CV_BGR2GRAY;
                    break;
                case 4:
                	code = CV_RGBA2GRAY;
                	break;
            }
            if (code < 0) {
            	convertedImage.close();
                throw new IOException("Cannot convert from " + image.channels()
                                                    + " to " + 1 + " channel.");
            }
            cvtColor(image, convertedImage, code);
        }
        
		List<List<SortedFrame>> framesPerLevel = null;
		List<SortedFrame> frames = null;
		
		start = System.currentTimeMillis();
		framesPerLevel = stage1(convertedImage, minFaceSize, maxFaceSize);
		stop = System.currentTimeMillis();
		log.info("Stage 1 took: " + (stop-start) + " ms.");
		
		start = System.currentTimeMillis();
		frames = stage2(convertedImage, framesPerLevel);
		stop = System.currentTimeMillis();
		log.info("Stage 2 took: " + (stop-start) + " ms.");

		start = System.currentTimeMillis();
		frames = stage3(convertedImage, frames);
		stop = System.currentTimeMillis();
		log.info("Stage 3 took: " + (stop-start) + " ms.");
		
		start = System.currentTimeMillis();
		frames = classify(convertedImage, frames);
		stop = System.currentTimeMillis();
		log.info("Classification took: " + (stop-start) + " ms.");

		convertedImage.close();
		return frames;

	}
	
	
	private List<List<SortedFrame>> stage1(Mat image, int minFaceSize, int maxFaceSize) {
		
		if (image == null) {
			return null;
		}
		
		int imWidth = image.cols();
		int imHeight = image.rows();
		
		if (imWidth<minFaceSize)
			return null;
		
		if (imHeight<minFaceSize)
			return null;

		int scale = 0;
		int kernelSize, stride;
		
		kernelSize = (int) (minFaceSize*Math.pow(zoomFactor, scale));
		stride = (int) ((double)stage1Stride/stage1KernelSize*kernelSize);

		List<List<SortedFrame>> allFrames = new ArrayList<List<SortedFrame>>(); 
		
		NativeImageLoader scaler = new NativeImageLoader(stage1KernelSize, stage1KernelSize, image.channels());
		

		int rows, cols;
		while (imWidth>=kernelSize && imHeight>=kernelSize && kernelSize<=maxFaceSize) {
			
			cols = (imWidth - kernelSize) / 2 / stride + 1;
			rows = (imHeight - kernelSize) / 2 / stride + 1;
			
			numOfFrames += (cols*rows)*4;

			List<SortedFrame> zoomLevelFrames = new ArrayList<SortedFrame>();
			List<DataSet> dsList = new ArrayList<DataSet>();
			
			Mat frame = null;
			for (int i=0; i<rows; i++) {
				for (int j=0; j<cols; j++) {
					
					frame = image.colRange(j*stride, j*stride+kernelSize)
							.rowRange(i*stride, i*stride+kernelSize);
				
					try {
						INDArray input = scaler.asMatrix(frame);
						//TODO: Riješiti trajno prilikom učitavanja (optimizovati).
						ipps.preProcess(input);
						dsList.add(new DataSet(input, Nd4j.create(new double[]{1.0, 0.0})));
						zoomLevelFrames.add(new SortedFrame(j*stride, i*stride, kernelSize));
					} catch (IOException e) {
						log.error("Error converting from Mat to INDArray!");
						e.printStackTrace();
					}
					
					frame.release();

					frame = image.colRange(imWidth - j*stride - kernelSize, imWidth - j*stride)
							.rowRange(i*stride, i*stride+kernelSize);
				
					try {
						INDArray input = scaler.asMatrix(frame);
						//TODO: Riješiti trajno prilikom učitavanja (optimizovati).
						ipps.preProcess(input);
						dsList.add(new DataSet(input, Nd4j.create(new double[]{1.0, 0.0})));
						zoomLevelFrames.add(new SortedFrame(imWidth - j*stride - kernelSize, i*stride, kernelSize));
					} catch (IOException e) {
						log.error("Error converting from Mat to INDArray!");
						e.printStackTrace();
					}

					frame.release();

					frame = image.colRange(j*stride, j*stride+kernelSize)
							.rowRange(imHeight - i*stride - kernelSize, imHeight - i*stride);
				
					try {
						INDArray input = scaler.asMatrix(frame);
						//TODO: Riješiti trajno prilikom učitavanja (optimizovati).
						ipps.preProcess(input);
						dsList.add(new DataSet(input, Nd4j.create(new double[]{1.0, 0.0})));
						zoomLevelFrames.add(new SortedFrame(j*stride, imHeight - i*stride - kernelSize, kernelSize));
					} catch (IOException e) {
						log.error("Error converting from Mat to INDArray!");
						e.printStackTrace();
					}

					frame.release();

					frame = image.colRange(imWidth - j*stride - kernelSize, imWidth - j*stride)
							.rowRange(imHeight - i*stride - kernelSize, imHeight - i*stride);
				
					try {
						INDArray input = scaler.asMatrix(frame);
						//TODO: Riješiti trajno prilikom učitavanja (optimizovati).
						ipps.preProcess(input);
						dsList.add(new DataSet(input, Nd4j.create(new double[]{1.0, 0.0})));
						zoomLevelFrames.add(new SortedFrame(imWidth - j*stride - kernelSize, imHeight - i*stride - kernelSize, kernelSize));
					} catch (IOException e) {
						log.error("Error converting from Mat to INDArray!");
						e.printStackTrace();
					}

					frame.release();
 
				}
			}
			
			if (!dsList.isEmpty()) {
				
				List<DataSet> batch = null;
				int cursor = 0;
				DataSet ds = null;
				INDArray result = null;
				
				List<SortedFrame> tmpList = new ArrayList<SortedFrame>();
				SortedFrame tmpFrame = null;
				List<DataSet> tmpDsList = new ArrayList<DataSet>();
				
				while (cursor<dsList.size()) {
					
					batch = dsList.subList(cursor, (cursor+batchSize>dsList.size() ? dsList.size() : cursor+batchSize));
					ds = DataSet.merge(batch);
					result = net13detection.output(ds.getFeatures());
					
					for (int i=0; i < result.rows(); i++) {
						if (result.getDouble(i, 1)>=thresholdT1) {
							tmpFrame = zoomLevelFrames.get(cursor+i);
							tmpFrame.setScore(result.getDouble(i, 1));
							tmpList.add(tmpFrame);
							tmpDsList.add(dsList.get(cursor+i));
						}
					}
					
					cursor += batchSize;
					
				}
				
				zoomLevelFrames = tmpList;
				dsList = tmpDsList;
				

				/*
				 * Calibrate remaining frames.
				 */
				
				if (!zoomLevelFrames.isEmpty()) {
				
					cursor = 0;
					int newX, newY, newW;
					
					while (cursor<dsList.size()) {
						
						batch = dsList.subList(cursor, (cursor+batchSize>dsList.size() ? dsList.size() : cursor+batchSize));
						ds = DataSet.merge(batch);
						result = net13calibration.output(ds.getFeatures());
						
						for (int i=0; i < result.rows(); i++) {
							
							double sum = 0.0;
							for (int k=0; k<45; k++) {
								if (result.getDouble(i, k)>=calibrationThreshold) {
									sum += result.getDouble(i, k);
								}
							}
							
							double sn=0.0, xn=0.0, yn=0.0;
							for (int k=0; k<45; k++) {
								if (result.getDouble(i, k)>=calibrationThreshold) {
									sn = sn + result.getDouble(i, k)/sum * calibrationParams[k][0];
									xn = xn + result.getDouble(i, k)/sum * calibrationParams[k][1];
									yn = yn + result.getDouble(i, k)/sum * calibrationParams[k][2];
								}
							}
							
							tmpFrame = zoomLevelFrames.get(cursor+i);
							
							newX=tmpFrame.x+(int)(xn*tmpFrame.w);
							newY=tmpFrame.y+(int)(yn*tmpFrame.w);
							newW=(int)(Math.pow(1.1, sn)*tmpFrame.w);
					
							if (newX>=0 && newY>=0 &&
									newX+newW<=image.cols() && newY+newW<=image.rows()) {
								tmpFrame.x = newX;
								tmpFrame.y = newY;
								tmpFrame.w = newW;
							}
							
						}
						
						cursor += batchSize;

					}

					
					/*
					 * Filter overlapped frames.
					 */
					SortedSet<SortedFrame> sortedFrames = new TreeSet<SortedFrame>(new SortedFrameComparator());
					sortedFrames.addAll(zoomLevelFrames);
					
					zoomLevelFrames.clear();
					tmpList = new ArrayList<SortedFrame>();
					
					SortedFrame bestFrame = null;
					while (sortedFrames.size()>0) {
						if (sortedFrames.size()==1) {
							zoomLevelFrames.add(sortedFrames.first());
							break;
						}
						bestFrame = sortedFrames.first();
						sortedFrames.remove(bestFrame);
						zoomLevelFrames.add(bestFrame);
						tmpList.clear();
						for (SortedFrame f : sortedFrames) {
							if (iOu(bestFrame, f)>minIoU) {
								tmpList.add(f);
							}
						}
						sortedFrames.removeAll(tmpList);
					}
					
					
					if (zoomLevelFrames.size()>0) {
						allFrames.add(zoomLevelFrames);
					}

				}
			}

			scale ++;
			kernelSize = (int) (minFaceSize*Math.pow(zoomFactor, scale));
			stride = (int) ((double)stage1Stride/stage1KernelSize*kernelSize);
		}

		return allFrames;
		
	}
	
	
	private List<SortedFrame> stage2(Mat image, List<List<SortedFrame>> frames) {
		
		if (image == null || frames == null) {
			return null;
		}
		
		NativeImageLoader scaler = new NativeImageLoader(stage2KernelSize, stage2KernelSize, image.channels());
		
		List<SortedFrame> allFrames = new ArrayList<SortedFrame>();
		Mat frame = null;
		INDArray input, result;
		
		List<DataSet> dsList = new ArrayList<DataSet>();
		
		for (List<SortedFrame> zoomLevelFrames:frames) {
			
			dsList.clear();
			
			for (SortedFrame sf:zoomLevelFrames) {
				
				frame = image.colRange(sf.x, sf.x+sf.w)
						.rowRange(sf.y, sf.y+sf.w);
			
				try {
					input = scaler.asMatrix(frame);
					ipps.preProcess(input);
					dsList.add(new DataSet(input, Nd4j.create(new double[]{1.0, 0.0})));
				} catch (IOException e) {
					log.error("Error converting from Mat to INDArray!");
					e.printStackTrace();
				}

				frame.release();
				
			}
			
			int cursor = 0;
			List<DataSet> batch = null;
			List<SortedFrame> tmpList = new ArrayList<SortedFrame>();
			List<DataSet> tmpDsList = new ArrayList<DataSet>();
			DataSet ds = null;
			SortedFrame tmpFrame = null;
			
			while (cursor<dsList.size()) {
				
				batch = dsList.subList(cursor, (cursor+batchSize>dsList.size() ? dsList.size() : cursor+batchSize));
				ds = DataSet.merge(batch);
				result = net25detection.output(ds.getFeatures());
				for (int i=0; i<result.rows(); i++) {
					if (result.getDouble(i, 1)>=thresholdT2) {
						tmpFrame = zoomLevelFrames.get(cursor+i);
						tmpFrame.setScore(result.getDouble(i, 1));
						tmpList.add(tmpFrame);
						tmpDsList.add(batch.get(i));
					}
				}
				
				cursor += batchSize;
				
			}
			
			zoomLevelFrames = tmpList;
			dsList = tmpDsList;
			

			/*
			 * Calibrate remaining frames.
			 */
			
			if (!zoomLevelFrames.isEmpty()) {
			
				cursor = 0;
				int newX, newY, newW;
				
				while (cursor<dsList.size()) {
					
					batch = dsList.subList(cursor, (cursor+batchSize>dsList.size() ? dsList.size() : cursor+batchSize));
					ds = DataSet.merge(batch);
					result = net25calibration.output(ds.getFeatures());
					
					for (int i=0; i < result.rows(); i++) {
						
						double sum = 0.0;
						for (int k=0; k<45; k++) {
							if (result.getDouble(i, k)>=calibrationThreshold) {
								sum += result.getDouble(i, k);
							}
						}
						
						double sn=0.0, xn=0.0, yn=0.0;
						for (int k=0; k<45; k++) {
							if (result.getDouble(i, k)>=calibrationThreshold) {
								sn = sn + result.getDouble(i, k)/sum * calibrationParams[k][0];
								xn = xn + result.getDouble(i, k)/sum * calibrationParams[k][1];
								yn = yn + result.getDouble(i, k)/sum * calibrationParams[k][2];
							}
						}
						
						tmpFrame = zoomLevelFrames.get(cursor+i);
						
						newX=tmpFrame.x+(int)(xn*tmpFrame.w);
						newY=tmpFrame.y+(int)(yn*tmpFrame.w);
						newW=(int)(Math.pow(1.1, sn)*tmpFrame.w);
				
						if (newX>=0 && newY>=0 &&
								newX+newW<=image.cols() && newY+newW<=image.rows()) {
							tmpFrame.x = newX;
							tmpFrame.y = newY;
							tmpFrame.w = newW;
						}
						
					}
					
					cursor += batchSize;

				}

				
				/*
				 * Filter overlapped frames.
				 */
				SortedSet<SortedFrame> sortedFrames = new TreeSet<SortedFrame>(new SortedFrameComparator());
				sortedFrames.addAll(zoomLevelFrames);
				
				zoomLevelFrames.clear();
				tmpList = new ArrayList<SortedFrame>();
				
				SortedFrame bestFrame = null;
				while (sortedFrames.size()>0) {
					if (sortedFrames.size()==1) {
						zoomLevelFrames.add(sortedFrames.first());
						break;
					}
					bestFrame = sortedFrames.first();
					sortedFrames.remove(bestFrame);
					zoomLevelFrames.add(bestFrame);
					tmpList.clear();
					for (SortedFrame f : sortedFrames) {
						if (iOu(bestFrame, f)>minIoU) {
							tmpList.add(f);
						}
					}
					sortedFrames.removeAll(tmpList);
				}
				
				
				if (zoomLevelFrames.size()>0) {
					allFrames.addAll(zoomLevelFrames);
				}

			}
			
		}
		
		return allFrames;
	}
	
	
	private List<SortedFrame> stage3(Mat image, List<SortedFrame> frames) {
		
		if (image == null || frames == null) {
			return null;
		}
		
		NativeImageLoader scaler = new NativeImageLoader(stage3KernelSize, stage3KernelSize, image.channels());
		
		List<SortedFrame> allFrames = new ArrayList<SortedFrame>();
		List<DataSet> dsList = new ArrayList<DataSet>();
		Mat frame = null;
		INDArray input, result;
		
		for (SortedFrame f:frames) {
			try {
				frame = image.colRange(f.x, f.x+f.w).rowRange(f.y, f.y+f.w);
				input = scaler.asMatrix(frame);
				ipps.preProcess(input);
				dsList.add(new DataSet(input, Nd4j.create(new double[]{1.0, 0.0})));
			} catch (IOException e) {
				log.error("Error converting from Mat to INDArray!");
				e.printStackTrace();
			}
		}
		
		int cursor = 0;
		List<DataSet> batch = null;
		DataSet ds = null;
		List<DataSet> tmpDsList = new ArrayList<DataSet>();
		List<SortedFrame> tmpList = new ArrayList<SortedFrame>();
		SortedFrame tmpFrame = null;
		
		while (cursor<dsList.size()) {
			
			batch = dsList.subList(cursor, (cursor+batchSize>dsList.size() ? dsList.size() : cursor+batchSize));
			ds = DataSet.merge(batch);
			result = net51detection.output(ds.getFeatures());
			
			for (int i=0; i<result.rows(); i++) {
				if (result.getDouble(i, 1)>=thresholdT3) {
					tmpFrame = frames.get(cursor + i);
					tmpFrame.setScore(result.getDouble(i, 1));
					tmpList.add(tmpFrame);
					tmpDsList.add(batch.get(i));
				}
			}
			
			cursor += batchSize;
		}
		
		frames = tmpList;
		dsList = tmpDsList;
		

		/*
		 * Calibrate remaining frames.
		 */
		
		if (!frames.isEmpty()) {
		
			cursor = 0;
			int newX, newY, newW;
			
			while (cursor<dsList.size()) {
				
				batch = dsList.subList(cursor, (cursor+batchSize>dsList.size() ? dsList.size() : cursor+batchSize));
				ds = DataSet.merge(batch);
				result = net51calibration.output(ds.getFeatures());
				
				for (int i=0; i < result.rows(); i++) {
					
					double sum = 0.0;
					for (int k=0; k<45; k++) {
						if (result.getDouble(i, k)>=calibrationThreshold) {
							sum += result.getDouble(i, k);
						}
					}
					
					double sn=0.0, xn=0.0, yn=0.0;
					for (int k=0; k<45; k++) {
						if (result.getDouble(i, k)>=calibrationThreshold) {
							sn = sn + result.getDouble(i, k)/sum * calibrationParams[k][0];
							xn = xn + result.getDouble(i, k)/sum * calibrationParams[k][1];
							yn = yn + result.getDouble(i, k)/sum * calibrationParams[k][2];
						}
					}
					
					tmpFrame = frames.get(cursor+i);
					
					newX=tmpFrame.x+(int)(xn*tmpFrame.w);
					newY=tmpFrame.y+(int)(yn*tmpFrame.w);
					newW=(int)(Math.pow(1.1, sn)*tmpFrame.w);
			
					if (newX>=0 && newY>=0 &&
							newX+newW<=image.cols() && newY+newW<=image.rows()) {
						tmpFrame.x = newX;
						tmpFrame.y = newY;
						tmpFrame.w = newW;
					}
					
				}
				
				cursor += batchSize;

			}

			
			/*
			 * Filter overlapped frames.
			 */
			SortedSet<SortedFrame> sortedFrames = new TreeSet<SortedFrame>(new SortedFrameComparator());
			sortedFrames.addAll(frames);
			
			frames.clear();
			tmpList = new ArrayList<SortedFrame>();
			
			SortedFrame bestFrame = null;
			while (sortedFrames.size()>0) {
				if (sortedFrames.size()==1) {
					frames.add(sortedFrames.first());
					break;
				}
				bestFrame = sortedFrames.first();
				sortedFrames.remove(bestFrame);
				frames.add(bestFrame);
				tmpList.clear();
				for (SortedFrame f : sortedFrames) {
					if (iOu(bestFrame, f)>minIoU) {
						tmpList.add(f);
					}
				}
				sortedFrames.removeAll(tmpList);
			}
			
			
			if (frames.size()>0) {
				allFrames.addAll(frames);
			}
		
		}
		
		return allFrames;

	}
	
	
	public static double iOu(Frame f1, Frame f2) {
		
		int intersanction = 0;
		int union = 0;
		
		intersanction =
				Math.max(0, (Math.min(f1.x+f1.w, f2.x+f2.w) - Math.max(f1.x, f2.x))) *
				Math.max(0, (Math.min(f1.y+f1.w, f2.y+f2.w) - Math.max(f1.y, f2.y)));
		
//		union = f1.w * f1.w + f2.w * f2.w - intersanction;
		union = Math.min(f1.w * f1.w, f2.w * f2.w);
		
		return (double)intersanction/union;
	}
	
	
	private List<SortedFrame> classify(Mat image, List<SortedFrame> frames) {
		
		if (image == null || frames == null) {
			return null;
		}
		
		NativeImageLoader scaler = new NativeImageLoader(stage3KernelSize, stage3KernelSize, image.channels());
		
		List<SortedFrame> allFrames = new ArrayList<SortedFrame>();
		List<DataSet> dsList = new ArrayList<DataSet>();
		Mat frame = null;
		INDArray input, result;
		
		int w, x, y;
		double dx, dy, dw;
		for (SortedFrame f:frames) {
			try {
//				dw=1.1*1.1*f.w;
//				dx=f.x-(dw-(double)f.w/2.0);
//				dy=f.y-(dw-(double)f.w/2.0);
//				w=(int)dw;
//				x=(int)dx;
//				if (x<0) x=0;
//				y=(int)dy;
//				if (y<0) y=0;
				x=f.x;
				y=f.y;
				w=f.w;
				frame = image.colRange(x, x+w).rowRange(y, y+w);
				input = scaler.asMatrix(frame);
				ipps.preProcess(input);
				dsList.add(new DataSet(input, Nd4j.create(new double[]{1.0, 0.0})));
			} catch (IOException e) {
				log.error("Error converting from Mat to INDArray!");
				e.printStackTrace();
			}
		}
		
		int cursor = 0;
		List<DataSet> batch = null;
		DataSet ds = null;
		SortedFrame tmpFrame = null;
		
		while (cursor<dsList.size()) {
			
			batch = dsList.subList(cursor, (cursor+batchSize>dsList.size() ? dsList.size() : cursor+batchSize));
			ds = DataSet.merge(batch);
			result = classification.output(ds.getFeatures());
			
			for (int i=0; i<result.rows(); i++) {
				tmpFrame = frames.get(cursor + i);
				tmpFrame.setScore(result.getDouble(i, 1));
				allFrames.add(tmpFrame);
			}
			
			cursor += batchSize;
		}
		
		return allFrames;

	}
	

	public static void main(String[] args) throws IOException {
		
//		testDetection(args);
		
//		negativeSampleMining(args);
		
//		detectFaces(args);
		
		detectAndClassifyFaces(args);
		
	}
	
	
	public static void testDetection(String[] args) throws IOException {
		
//		File srcFolder = new File("/home/ambra/Deep Learning/FDDB/originalPics/");
		File srcFolder = new File("/home/ambra/tttt/");
		File dstFolder = new File("/home/ambra/Deep Learning/FDDB/detected/");
		
		PrintStream output = new PrintStream(new FileOutputStream("/home/ambra/Deep Learning/FDDB/stage3.txt"), true);
		
		MultiLayerNetwork net13detection = null;
		MultiLayerNetwork net25detection = null;
		MultiLayerNetwork net51detection = null;
		MultiLayerNetwork net13calibration = null;
		MultiLayerNetwork net25calibration = null;
		MultiLayerNetwork net51calibration = null;
		MultiLayerNetwork classification = null;
		
		try {
			
			net13detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-13/9943.zip", false);
			net25detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-25/9986.zip", false);
			net51detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-51/9986.zip", false);
			net13calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-13/7095.zip", false);
			net25calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-25/7963.zip", false);
			net51calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-51/9112.zip", false);
			classification =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/classification/9439.zip", false);
			
			
			
		} catch (IOException err) {
			log.error("Unable to initialize the networks!");
			err.printStackTrace();
			System.exit(1);
		}
		
		FullCascade4 fc1 = new FullCascade4(
				net13detection,
				net13calibration,
				net25detection,
				net25calibration,
				net51detection,
				net51calibration,
				classification);
		
		long numOfFaces = 0;
		
		Mat srcImage = new Mat();

		List<SortedFrame> detectedFrames = null;
		
		try (BufferedReader in = new BufferedReader(
				new InputStreamReader(
						new FileInputStream("/home/ambra/Deep Learning/FDDB/FDDB-folds/FDDB-all.txt")))) {
			String line = null;
			while ((line = in.readLine())!=null) {
				srcImage = imread(srcFolder.getAbsolutePath() + File.separator + line + ".jpg");
		        log.info("Processing " + line);
				detectedFrames = fc1.detectFaces(srcImage, 40, Integer.MAX_VALUE);
				if (detectedFrames.size()>0) {
					numOfFaces = numOfFaces + detectedFrames.size();
					log.info(detectedFrames.size() + " faces detected!");
					output.println(line);
					output.println(detectedFrames.size());
					for (SortedFrame f:detectedFrames) {
						output.printf("%d %d %d %d %f\n", f.x, f.y-(int)(0.2*f.w), f.w, (int)(1.2*f.w), f.getScore());
						if (f.getScore()<0.5) {
							rectangle(srcImage, new Rect(f.x, f.y, f.w, f.w), new Scalar(0, 0, 255.0, 0));
						}
					}
					for (SortedFrame f:detectedFrames) {
						if (f.getScore()>=0.5) {
							rectangle(srcImage, new Rect(f.x, f.y, f.w, f.w), new Scalar(0, 255.0, 0, 0));
						}
					}
				} else {
					output.println(line);
					output.println("0");
					log.warn("0 faces detected!");
				}
				line = line.replace("/", "-");
				//imwrite(dstFolder + File.separator + line + ".jpg", srcImage);
				srcImage.release();
				log.info(String.format("Rejected: %.2f%%", (1-(double)numOfFaces/numOfFrames)*100));
			}
			
			log.info("Total frames: " + numOfFrames);
			log.info("Total faces: " + numOfFaces);
		} catch (IOException err) {
			log.error("Failure opening the file!");
			err.printStackTrace();
		}
		
		output.close();
		
	}
	
	
	public static void negativeSampleMining(String[] args) throws IOException {
		
		File srcFolder = new File("/home/ambra/Deep Learning/ImageNet/class2/");
		File dstFolder = new File("/home/ambra/Deep Learning/ImageNet/detected/");
		
		MultiLayerNetwork net13detection = null;
		MultiLayerNetwork net25detection = null;
		MultiLayerNetwork net51detection = null;
		MultiLayerNetwork net13calibration = null;
		MultiLayerNetwork net25calibration = null;
		MultiLayerNetwork net51calibration = null;
		MultiLayerNetwork classification = null;

		try {
			
			net13detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-13/9943.zip", false);
			net25detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-25/9986.zip", false);
			net51detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-51/9986.zip", false);
			net13calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-13/7095.zip", false);
			net25calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-25/7963.zip", false);
			net51calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-51/9112.zip", false);
			classification =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/classification/9439.zip", false);
			
		} catch (IOException err) {
			log.error("Unable to initialize the networks!");
			err.printStackTrace();
			System.exit(1);
		}
		
		FullCascade4 fc1 = new FullCascade4(
				net13detection,
				net13calibration,
				net25detection,
				net25calibration,
				net51detection,
				net51calibration,
				classification);
		
		long numOfFaces = 0;
		
		long start, stop;
		List<SortedFrame> detectedFrames = null;
		
		File[] files = srcFolder.listFiles();
		
		long fileCount = 0;
		
		for (File file:files) {
			
			fileCount++;
			log.info("Files processed: " + fileCount);
			
//			if (file.getName().equals("00014.jpg"))	continue;
//			if (file.getName().equals("00142.jpg"))	continue;
//			if (file.getName().equals("06071.jpg"))	continue;
//			if (file.getName().equals("07001.jpg"))	continue;
//			if (file.getName().equals("07522.jpg"))	continue;
//			if (file.getName().equals("07592.jpg"))	continue;
//			if (file.getName().equals("09140.jpg"))	continue;
//			if (file.getName().equals("09339.jpg"))	continue;
//			if (file.getName().equals("09504.jpg"))	continue;
//			if (file.getName().equals("09600.jpg"))	continue;
//			if (file.getName().equals("09716.jpg"))	continue;
//			if (file.getName().equals("09808.jpg"))	continue;
//			if (file.getName().equals("09973.jpg"))	continue;
//			if (file.getName().equals("09974.jpg"))	continue;
//			if (file.getName().equals("11030.jpg"))	continue;
			
			if (new File(dstFolder.getAbsolutePath() + File.separator + 
					file.getName().substring(0, file.getName().lastIndexOf(".")) + "-1.png").exists()) {
				continue;
			}
			
			Mat srcImage = new Mat();
			srcImage = imread(file.getAbsolutePath());
	        log.info("Processing " + file.getName());
			detectedFrames = fc1.detectFaces(srcImage, 51, Integer.MAX_VALUE);
			if (detectedFrames!=null && detectedFrames.size()>0) {
				numOfFaces = numOfFaces + detectedFrames.size();
				log.info(detectedFrames.size() + " faces detected!");

				Mat frame = new Mat();
				
				int count=0;
				for (SortedFrame f:detectedFrames) {
					frame = srcImage.colRange(f.x, f.x+f.w)
							.rowRange(f.y, f.y+f.w);
					
			        if (frame.channels() != 1) {
			            int code = -1;
			            switch (frame.channels()) {
			                case 3:
			                    code = CV_BGR2GRAY;
			                    break;
			                case 4:
			                	code = CV_RGBA2GRAY;
			                	break;
			            }
			            if (code < 0) {
			                throw new IOException("Cannot convert from " + frame.channels()
			                                                    + " to " + 1 + " channel.");
			            }
			            cvtColor(frame, frame, code);
			        }
			        
			        resize(frame, frame, new Size(51, 51));
		            count++;
		            imwrite(dstFolder.getAbsolutePath() + File.separator +
		            		file.getName().substring(0, file.getName().lastIndexOf(".")) + 
		            				"-" + count + ".png", frame);
			        frame.close();
					
				}
			} else {
				log.warn("0 faces detected!");
			}
			
			srcImage.close();
		}
		
	}
	
	public static void detectFaces(String[] args) throws IOException {
		
		
		File srcFolder = new File("/home/ambra/tttt/");
		File dstFolder = new File("/home/ambra/Deep Learning/ImageNet/detected/");
		
		MultiLayerNetwork net13detection = null;
		MultiLayerNetwork net25detection = null;
		MultiLayerNetwork net51detection = null;
		MultiLayerNetwork net13calibration = null;
		MultiLayerNetwork net25calibration = null;
		MultiLayerNetwork net51calibration = null;
		MultiLayerNetwork classification = null;

		try {
			
			net13detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-13/9943.zip", false);
			net25detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-25/9986.zip", false);
			net51detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-51/9986.zip", false);
			net13calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-13/7095.zip", false);
			net25calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-25/7963.zip", false);
			net51calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-51/9112.zip", false);
			classification =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/classification/9439.zip", false);
			
		} catch (IOException err) {
			log.error("Unable to initialize the networks!");
			err.printStackTrace();
			System.exit(1);
		}
		
		FullCascade4 fc1 = new FullCascade4(
				net13detection,
				net13calibration,
				net25detection,
				net25calibration,
				net51detection,
				net51calibration,
				classification);
		
		long numOfFaces = 0;
		
		Mat srcImage = new Mat();

		List<SortedFrame> detectedFrames = null;

		
		File[] files = srcFolder.listFiles();
		
		for (File f:files) {
			
			srcImage = imread(f.getAbsolutePath());
	        log.info("Processing " + f.getName());
			detectedFrames = fc1.detectFaces(srcImage, 40, Integer.MAX_VALUE);
			if (detectedFrames.size()>0) {
				numOfFaces = numOfFaces + detectedFrames.size();
				log.info(detectedFrames.size() + " faces detected!");

				for (SortedFrame sf:detectedFrames) {
					if (sf.getScore()<0.5) {
						rectangle(srcImage, new Rect(sf.x, sf.y, sf.w, sf.w), new Scalar(0, 0, 255.0, 0));
					} else {
						rectangle(srcImage, new Rect(sf.x, sf.y, sf.w, sf.w), new Scalar(0, 255.0, 0, 0));
					}
				}

			} else {
				log.warn("0 faces detected!");
			}
			imwrite(dstFolder + File.separator + f.getName(), srcImage);
			srcImage.release();
		}
		
	}

	public static void detectAndClassifyFaces(String[] args) throws IOException {
		
		
		MultiLayerNetwork net13detection = null;
		MultiLayerNetwork net25detection = null;
		MultiLayerNetwork net51detection = null;
		MultiLayerNetwork net13calibration = null;
		MultiLayerNetwork net25calibration = null;
		MultiLayerNetwork net51calibration = null;
		MultiLayerNetwork classification = null;

		try {
			
			net13detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-13/9943.zip", false);
			net25detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-25/9986.zip", false);
			net51detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-51/9986.zip", false);
			net13calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-13/7095.zip", false);
			net25calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-25/7963.zip", false);
			net51calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-51/9112.zip", false);
			classification =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/model.zip", false);
			
		} catch (IOException err) {
			log.error("Unable to initialize the networks!");
			err.printStackTrace();
			System.exit(1);
		}
		
		FullCascade4 fc1 = new FullCascade4(
				net13detection,
				net13calibration,
				net25detection,
				net25calibration,
				net51detection,
				net51calibration,
				classification);
		
		long numOfFaces = 0;
		
		Mat srcImage = new Mat();

		List<SortedFrame> detectedFrames = null;

		
		VideoCapture video = new VideoCapture();
		boolean OK = video.open(0+CAP_V4L2);
		OK = video.grab();
		Mat image = new Mat();
		Mat gray = new Mat();
		namedWindow("video", CV_WINDOW_AUTOSIZE);
		startWindowThread();
		while (true) {
			video.read(image);

			detectedFrames = fc1.detectFaces(image, 40, Integer.MAX_VALUE);

			if (detectedFrames != null && detectedFrames.size()>0) {

				log.info(detectedFrames.size() + " faces detected!");
	
				int percentage;
				for (SortedFrame sf:detectedFrames) {
					if (sf.getScore()>=fc1.classificationThreshold) {
						percentage = (int) ((sf.getScore()-fc1.classificationThreshold)/(1.0-fc1.classificationThreshold)*sf.w);
						rectangle(image, new Rect(sf.x, sf.y, sf.w, sf.w), new Scalar(255, 0, 0, 0));
						line(image, new Point(sf.x, sf.y), new Point(sf.x+percentage, sf.y), new Scalar(255, 0, 0, 0), 3, 0, 0);
						line(image, new Point(sf.x, sf.y+sf.w), new Point(sf.x+percentage, sf.y+sf.w), new Scalar(255, 0, 0, 0), 3, 0, 0);
					} else {
						percentage = (int) ((fc1.classificationThreshold-sf.getScore())/fc1.classificationThreshold*sf.w);
						rectangle(image, new Rect(sf.x, sf.y, sf.w, sf.w), new Scalar(0, 0, 255, 0));
						line(image, new Point(sf.x, sf.y), new Point(sf.x+percentage, sf.y), new Scalar(0, 0, 255, 0), 3, 0, 0);
						line(image, new Point(sf.x, sf.y+sf.w), new Point(sf.x+percentage, sf.y+sf.w), new Scalar(0, 0, 255, 0), 3, 0, 0);
					}
				}

			} else {
				log.warn("0 faces detected!");
			}
	        
			imshow("video", image);
			
			
		}
		
		
		
//		File[] files = srcFolder.listFiles();
//		
//		for (File f:files) {
//			
//			srcImage = imread(f.getAbsolutePath());
//	        log.info("Processing " + f.getName());
//			detectedFrames = fc1.detectFaces(srcImage, 40, Integer.MAX_VALUE);
//			if (detectedFrames.size()>0) {
//				numOfFaces = numOfFaces + detectedFrames.size();
//				log.info(detectedFrames.size() + " faces detected!");
//
//				int percentage;
//				for (SortedFrame sf:detectedFrames) {
//					if (sf.getScore()>=fc1.classificationThreshold) {
//						percentage = (int) ((sf.getScore()-fc1.classificationThreshold)/(1.0-fc1.classificationThreshold)*sf.w);
//						rectangle(srcImage, new Rect(sf.x, sf.y, sf.w, sf.w), new Scalar(255, 0, 0, 0));
//						line(srcImage, new Point(sf.x, sf.y), new Point(sf.x+percentage, sf.y), new Scalar(255, 0, 0, 0), 3, 0, 0);
//						line(srcImage, new Point(sf.x, sf.y+sf.w), new Point(sf.x+percentage, sf.y+sf.w), new Scalar(255, 0, 0, 0), 3, 0, 0);
//					} else {
//						percentage = (int) ((fc1.classificationThreshold-sf.getScore())/fc1.classificationThreshold*sf.w);
//						rectangle(srcImage, new Rect(sf.x, sf.y, sf.w, sf.w), new Scalar(0, 0, 255, 0));
//						line(srcImage, new Point(sf.x, sf.y), new Point(sf.x+percentage, sf.y), new Scalar(0, 0, 255, 0), 3, 0, 0);
//						line(srcImage, new Point(sf.x, sf.y+sf.w), new Point(sf.x+percentage, sf.y+sf.w), new Scalar(0, 0, 255, 0), 3, 0, 0);
//					}
//				}
//
//			} else {
//				log.warn("0 faces detected!");
//			}
//			imwrite(dstFolder + File.separator + f.getName(), srcImage);
//			srcImage.release();
//		}
		
	}

}
