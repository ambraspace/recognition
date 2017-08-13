package com.ambraspace.recognition;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2RGBA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_GRAY2BGR;
import static org.bytedeco.javacpp.opencv_imgproc.CV_GRAY2RGBA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RGBA2BGR;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RGBA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.SimpleLayout;
import org.apache.log4j.spi.Filter;
import org.apache.log4j.spi.LoggingEvent;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FullCascade1 {
	
	private static long numOfFrames = 0;
	
	private static final Logger log = LoggerFactory.getLogger(FullCascade1.class);
	
	private final MultiLayerNetwork net13detection;
	private final MultiLayerNetwork net25detection;
	private final MultiLayerNetwork net51detection;
	private final MultiLayerNetwork net13calibration;
	private final MultiLayerNetwork net25calibration;
	private final MultiLayerNetwork net51calibration;
	
	ImagePreProcessingScaler ipps;
	
	private final int stage1KernelSize = 13;
	private final int stage1Stride = 3;

	private final int stage2KernelSize = 25;

	private final int stage3KernelSize = 51;
	
	private final double zoomFactor = 1.46;

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

	double minIoU = 0.5;

	
	
	public FullCascade1(
			MultiLayerNetwork net13detection,
			MultiLayerNetwork net13calibration,
			MultiLayerNetwork net25detection,
			MultiLayerNetwork net25calibration,
			MultiLayerNetwork net51detection,
			MultiLayerNetwork net51calibration) {
		
		this.net13detection = net13detection;
		this.net25detection = net25detection;
		this.net51detection = net51detection;
		this.net13calibration = net13calibration;
		this.net25calibration = net25calibration;
		this.net51calibration = net51calibration;
		
		ipps = new ImagePreProcessingScaler();
		
		for (int i=0; i<calibrationParams.length; i++) {
			calibrationParams[i][1]=calibrationParams[i][1]+(1-Math.pow(1.1, calibrationParams[i][0]))/2.0;
			calibrationParams[i][2]=calibrationParams[i][2]+(1-Math.pow(1.1, calibrationParams[i][0]))/2.0;
		}
		
	}
	

	public List<SortedFrame> detectFaces(Mat image, int minFaceSize, int maxFaceSize) {
		
		List<List<SortedFrame>> framesPerLevel = null;
		List<SortedFrame> frames = null;
		
//		List<SortedFrame> frames = new ArrayList<SortedFrame>();
		
		framesPerLevel = stage1(image, minFaceSize, maxFaceSize);
		
//		for (List<SortedFrame> list : framesPerLevel) {
//			frames.addAll(list);
//		}
		frames = stage2(image, framesPerLevel);
//		frames = stage3(image, frames);
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
			
			cols = (imWidth - kernelSize) / stride + 1;
			rows = (imHeight - kernelSize) / stride + 1;
			
			numOfFrames += (cols*rows);

			List<SortedFrame> zoomLevelFrames = new ArrayList<SortedFrame>();
			
			Mat frame = null;
			int newX, newY, newW;
			for (int i=0; i<rows; i++) {
				for (int j=0; j<cols; j++) {
					
					frame = image.colRange(j*stride, j*stride+kernelSize)
							.rowRange(i*stride, i*stride+kernelSize);
				
					try {

						INDArray input = scaler.asMatrix(frame);
						ipps.preProcess(input);
						INDArray result = net13detection.output(input);

						if (result.getDouble(1)>=0.001) { //T1
							result = net13calibration.output(input);
							double sum = 0.0;
							for (int k=0; k<45; k++) {
								if (result.getDouble(k)>=0.2) {
									sum += result.getDouble(k);
								}
							}
							double sn=0.0, xn=0.0, yn=0.0;
							for (int k=0; k<45; k++) {
								if (result.getDouble(k)>=0.2) {
									sn = sn + result.getDouble(k)/sum * calibrationParams[k][0];
									xn = xn + result.getDouble(k)/sum * calibrationParams[k][1];
									yn = yn + result.getDouble(k)/sum * calibrationParams[k][2];
								}
							}
							newX=j*stride+(int)(xn*kernelSize);
							newY=i*stride+(int)(yn*kernelSize);
							newW=(int)(Math.pow(1.1, sn)*kernelSize);
							if (newX<0 || newY<0 ||
									newX+newW>image.cols() || newY+newW>image.rows()) {
								zoomLevelFrames.add(new SortedFrame(j*stride, i*stride, kernelSize));
							} else {
								zoomLevelFrames.add(new SortedFrame(newX, newY, newW));
							}
						}
					} catch (IOException e) {
						log.error("Error converting from Mat to INDArray!");
						e.printStackTrace();
					}

					frame.release();
				}
			}
			
			SortedSet<SortedFrame> sortedFrames = new TreeSet<SortedFrame>(new SortedFrameComparator());
			
			INDArray input, output;

			for (SortedFrame f:zoomLevelFrames) {
				try {
					frame = image.colRange(f.x, f.x+f.w).rowRange(f.y, f.y+f.w);
					input = scaler.asMatrix(frame);
					ipps.preProcess(input);
					output = net13detection.output(input);
					f.setScore(output.getDouble(1));
					sortedFrames.add(f);
				} catch (IOException e) {
					e.printStackTrace();
				}
				
				frame.release();
			}
			
			zoomLevelFrames.clear();
			List<Frame> group = new ArrayList<Frame>();

			SortedFrame bestFrame = null;
			while (sortedFrames.size()>0) {
				if (sortedFrames.size()==1) {
					zoomLevelFrames.add(sortedFrames.first());
					break;
				}
				bestFrame = sortedFrames.first();
				sortedFrames.remove(bestFrame);
				zoomLevelFrames.add(bestFrame);
				group.clear();
				for (SortedFrame f : sortedFrames) {
					if (iOu(bestFrame, f)>minIoU) {
						group.add(f);
					}
				}
				sortedFrames.removeAll(group);
			}
			
			
			if (zoomLevelFrames.size()>0) {
				allFrames.add(zoomLevelFrames);
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
		int newX, newY, newW;
		
		for (List<SortedFrame> zoomLevelFrames:frames) {
			
			List<SortedFrame> frameList = new ArrayList<SortedFrame>();
			
			for (SortedFrame f:zoomLevelFrames) {
				frame = image.colRange(f.x, f.x+f.w).rowRange(f.y, f.y+f.w);
				try {
					input = scaler.asMatrix(frame);
					ipps.preProcess(input);
					result = net25detection.output(input);
					if (result.getDouble(1)>=0.05) { //T2
						result = net25calibration.output(input);
						double sum = 0.0;
						for (int k=0; k<45; k++) {
							if (result.getDouble(k)>=0.2) {
								sum += result.getDouble(k);
							}
						}
						double sn=0.0, xn=0.0, yn=0.0;
						for (int k=0; k<45; k++) {
							if (result.getDouble(k)>=0.2) {
								sn = sn + result.getDouble(k)/sum * calibrationParams[k][0];
								xn = xn + result.getDouble(k)/sum * calibrationParams[k][1];
								yn = yn + result.getDouble(k)/sum * calibrationParams[k][2];
							}
						}
						newX=f.x+(int)(xn*f.w);
						newY=f.y+(int)(yn*f.w);
						newW=(int)(Math.pow(1.1, sn)*f.w);
						if (newX<0 || newY<0 ||
								newX+newW>image.cols() || newY+newW>image.rows()) {
							frameList.add(new SortedFrame(f.x, f.y, f.w));
						} else {
							frameList.add(new SortedFrame(newX, newY, newW));
						}
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			
			SortedSet<SortedFrame> sortedFrames = new TreeSet<SortedFrame>(new SortedFrameComparator());
	
			for (SortedFrame f:frameList) {
				frame = image.colRange(f.x, f.x+f.w).rowRange(f.y, f.y+f.w);
				try {
					input = scaler.asMatrix(frame);
					ipps.preProcess(input);
					result = net25detection.output(input);
					f.setScore(result.getDouble(1));
					sortedFrames.add(f);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			
			List<SortedFrame> group = new ArrayList<SortedFrame>();
	
			SortedFrame bestFrame = null;
			while (sortedFrames.size()>0) {
				if (sortedFrames.size()==1) {
					allFrames.add(sortedFrames.first());
					break;
				}
				bestFrame = sortedFrames.first();
				sortedFrames.remove(bestFrame);
				allFrames.add(bestFrame);
				group.clear();
				for (SortedFrame f : sortedFrames) {
					if (iOu(bestFrame, f)>minIoU) {
						group.add(f);
					}
				}
				sortedFrames.removeAll(group);
			}

		}
		
		return allFrames;
	}
	
	
	private List<Frame> stage3(Mat image, List<Frame> frames) {
		
		if (image == null || frames == null) {
			return null;
		}
		
		NativeImageLoader scaler = new NativeImageLoader(stage3KernelSize, stage3KernelSize, image.channels());
		
		List<Frame> allFrames = new ArrayList<Frame>();
		Mat frame = null;
		INDArray input, result;
		int newX, newY, newW;
		for (Frame f:frames) {
			try {
				frame = image.colRange(f.x, f.x+f.w).rowRange(f.y, f.y+f.w);
				input = scaler.asMatrix(frame);
				ipps.preProcess(input);
				result = net51detection.output(input);
				if (result.getDouble(1)>=0.5) {
					allFrames.add(f);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		SortedSet<SortedFrame> sortedFrames = new TreeSet<SortedFrame>(new SortedFrameComparator());

		SortedFrame sf = null;
		for (Frame f:allFrames) {
			try {
				frame = image.colRange(f.x, f.x+f.w).rowRange(f.y, f.y+f.w);
				input = scaler.asMatrix(frame);
				ipps.preProcess(input);
				result = net51detection.output(input);
				sf = new SortedFrame(f);
				sf.setScore(result.getDouble(1));
				sortedFrames.add(sf);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		allFrames.clear();
		List<Frame> group = new ArrayList<Frame>();

		Frame bestFrame = null;
		while (sortedFrames.size()>0) {
			if (sortedFrames.size()==1) {
				allFrames.add(sortedFrames.first());
				break;
			}
			bestFrame = sortedFrames.first();
			sortedFrames.remove(bestFrame);
			allFrames.add(bestFrame);
			group.clear();
			for (SortedFrame f : sortedFrames) {
				if (iOu(bestFrame, f)>minIoU) {
					group.add(f);
				}
			}
			sortedFrames.removeAll(group);
		}
		

		for (Frame f:allFrames) {
			frame = image.colRange(f.x, f.x+f.w).rowRange(f.y, f.y+f.w);
			try {
				input = scaler.asMatrix(frame);
				ipps.preProcess(input);
				result = net51calibration.output(input);
				double sum = 0.0;
				for (int k=0; k<45; k++) {
					if (result.getDouble(k)>=0.1) {
						sum += result.getDouble(k);
					}
				}
				double sn=0.0, xn=0.0, yn=0.0;
				for (int k=0; k<45; k++) {
					if (result.getDouble(k)>=0.1) {
						sn = sn + result.getDouble(k)/sum * calibrationParams[k][0];
						xn = xn + result.getDouble(k)/sum * calibrationParams[k][1];
						yn = yn + result.getDouble(k)/sum * calibrationParams[k][2];
					}
				}
				newX=f.x+(int)(xn*f.w);
				newY=f.y+(int)(yn*f.w);
				newW=(int)(Math.pow(1.1, sn)*f.w);
				if (newX<0 || newY<0 ||
						newX+newW>image.cols() || newY+newW>image.rows()) {
					allFrames.add(new Frame(f.x, f.y, f.w));
				} else {
					allFrames.add(new Frame(newX, newY, newW));
				}
			} catch (IOException e) {
				e.printStackTrace();
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
		
		union = f1.w * f1.w + f2.w * f2.w - intersanction;
		
		return (double)intersanction/union;
	}

	

	public static void main(String[] args) throws IOException {
		
		File srcFolder = new File("/home/ambra/Deep Learning/FDDB/originalPics/");
		File dstFolder = new File("/home/ambra/Deep Learning/FDDB/detected/");
		
		PrintStream output = new PrintStream(new FileOutputStream("/home/ambra/Deep Learning/FDDB/stage1.txt"), true);
		
		MultiLayerNetwork net13detection = null;
		MultiLayerNetwork net25detection = null;
		MultiLayerNetwork net51detection = null;
		MultiLayerNetwork net13calibration = null;
		MultiLayerNetwork net25calibration = null;
		MultiLayerNetwork net51calibration = null;

		try {
			
			net13detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-13/9943.zip", false);
			net25detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-25/9986.zip", false);
			net51detection =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/detection-51/9937.zip", false);
			net13calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-13/7095.zip", false);
			net25calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-25/7963.zip", false);
			net51calibration =
					ModelSerializer.restoreMultiLayerNetwork(
							"/home/ambra/Deep Learning/LFW/training/calibration-51/9112.zip", false);
			
		} catch (IOException err) {
			log.error("Unable to initialize the networks!");
			err.printStackTrace();
			System.exit(1);
		}
		
		FullCascade1 fc1 = new FullCascade1(
				net13detection,
				net13calibration,
				net25detection,
				net25calibration,
				net51detection,
				net51calibration);
		
		long numOfFaces = 0;
		
		Mat srcImage = new Mat(), srcGrayImage = new Mat();
		long start, stop;
		List<SortedFrame> detectedFrames = null;
		
		try (BufferedReader in = new BufferedReader(
				new InputStreamReader(
						new FileInputStream("/home/ambra/Deep Learning/FDDB/FDDB-folds/FDDB-all.txt")))) {
			String line = null;
			while ((line = in.readLine())!=null) {
				srcImage = imread(srcFolder.getAbsolutePath() + File.separator + line + ".jpg");
		        if (srcImage.channels() != 1) {
		            int code = -1;
		            switch (srcImage.channels()) {
		                case 3:
	                        code = CV_BGR2GRAY;
	                        break;
		                case 4:
		                	code = CV_RGBA2GRAY;
		                	break;
		            }
		            if (code < 0) {
		                throw new IOException("Cannot convert from " + srcImage.channels()
		                                                    + " to " + 1 + " channel.");
		            }
		            cvtColor(srcImage, srcGrayImage, code);
		        }
		        log.info("Processing " + line);
				start = System.currentTimeMillis();
				detectedFrames = fc1.detectFaces(srcGrayImage, 40, Integer.MAX_VALUE);
				stop = System.currentTimeMillis();
				log.info("It took " + (stop-start) + " ms.");
				if (detectedFrames.size()>0) {
					numOfFaces = numOfFaces + detectedFrames.size();
					log.info(detectedFrames.size() + " faces detected!");
					output.println(line);
					output.println(detectedFrames.size());
					int x, y, w;
					double cx, cy, dw;
					for (SortedFrame f:detectedFrames) {
						cx = f.x + (double)f.w/2;
						cy = f.y + (double)f.w/2;
						dw = (double)f.w * Math.pow(1.1, 3);
						x=(int)(cx-dw/2);
						y=(int)(cy-dw/2);
						w=(int)dw;
						output.printf("%d %d %d %d %f\n", x, y, w, w, f.getScore());
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
					log.warn("0 faces detected!");
				}
				line = line.replace("/", "-");
				imwrite(dstFolder + File.separator + line + ".jpg", srcImage);
				srcImage.release();
				srcGrayImage.release();
				log.info(String.format("Rejected: %.2f%%", (1-(double)numOfFaces/numOfFrames)*100));
			}
			
			log.info("Total frames: " + numOfFrames);
			log.info("Total faces: " + numOfFaces);
		} catch (IOException err) {
			log.error("Failure opening the file!");
			err.printStackTrace();
		}
		
	}

}
