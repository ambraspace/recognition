package com.ambraspace.recognition.dl4j;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.SimpleLayout;
import org.apache.log4j.spi.Filter;
import org.apache.log4j.spi.LoggingEvent;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Stage1 {
	
	private static final Logger log = LoggerFactory.getLogger(Stage1.class);
	
	private static long saveCounter = 0;
	
	private static MultiLayerNetwork detector;
	private static MultiLayerNetwork calibrator;
	private static ImagePreProcessingScaler ipps;
	private static double[][] scales = {
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
	
	private static class Frame {
		public final int x;
		public final int y;
		public final int w;
		
		public Frame(int x, int y, int w) {
			this.x = x;
			this.y = y;
			this.w = w;
		}
		
		@Override
		public String toString() {
			return String.format("x: %d, y: %d, w: %d", x, y, w);
		}

	}
	
	public static void main(String[] args) {
		
	
		try {
			detector = ModelSerializer.restoreMultiLayerNetwork("/home/ambra/Deep Learning/LFW/detection-13/9906.zip", false);
			calibrator = ModelSerializer.restoreMultiLayerNetwork("/home/ambra/Deep Learning/LFW/calibration-13/6942.zip", false);
			ipps = new ImagePreProcessingScaler();
		} catch (IOException e) {
			log.error("Error loading networks!");
			System.exit(1);
		}
		
		
		for (int i=0; i<scales.length; i++) {
			scales[i][1]=scales[i][1]+(1-Math.pow(1.1, scales[i][0]))/2.0;
			scales[i][2]=scales[i][2]+(1-Math.pow(1.1, scales[i][0]))/2.0;
		}
		
		
		int minFaceSize = 25;

		int kernelSize = 13;
		int stride = 3;

		
		File currentFile = null;
		Mat image = null;
		int imWidth, imHeight;
		int newStride, newKernelSize;
		
		
		File srcDir = new File("/home/ambra/Deep Learning/ImageNet/class1/");

		File[] files = srcDir.listFiles(new FileFilter() {
			@Override
			public boolean accept(File pathname) {
				if (pathname.getName().endsWith(".jpg")) {
					return true;
				} else {
					return false;
				}
			}
		});
		
		long fileCounter = 0;
		for (File file:files) {
			
			currentFile = file;
			if (!currentFile.exists()) {
				log.warn("File " + currentFile.getAbsolutePath() +
						" doesn't exist! Skipping.");
				continue;
			}
			
			image = imread(file.getAbsolutePath());
			imWidth = image.cols();
			imHeight = image.rows();
			
			if (imWidth<minFaceSize)
				continue;
			
			if (imHeight<minFaceSize)
				continue;

			int scale = 0;
			newKernelSize = (int) (minFaceSize*Math.pow(1.3, scale));
			newStride = (int) ((double)stride/kernelSize*newKernelSize);
			List<Frame> frames;
			while (imWidth>=newKernelSize && imHeight>=newKernelSize) {
				frames = collectFrames(image, newKernelSize, newStride);
				saveFrames(image, frames, file.getName());
				scale ++;
				newKernelSize = (int) (minFaceSize*Math.pow(1.3, scale));
				newStride = (int) ((double)stride/kernelSize*newKernelSize);
			}
			
			image.release();
			
			log.info(fileCounter + ": " + file.getName());
			fileCounter++;
		}
		
	}
	
	private static List<Frame> collectFrames(Mat image, int kernelSize, int stride) {

		Mat frame = null;
		int cols, rows;
		int imWidth, imHeight;
		imWidth = image.cols();
		imHeight = image.rows();
		
		
		cols = (imWidth-kernelSize)/stride +1;
		rows = (imHeight-kernelSize)/stride +1;

		List<Frame> frames = new ArrayList<Frame>();
		
		int newX, newY, newW;
		for (int i=0; i<rows; i++) {
			for (int j=0; j<cols; j++) {
				
				frame = image
						.colRange(j*stride, j*stride+kernelSize)
						.rowRange(i*stride, i*stride+kernelSize);
			
				try {
					INDArray input = new NativeImageLoader(13, 13, frame.channels()).asMatrix(frame);
					ipps.preProcess(input);
					INDArray result = detector.output(input);
					if (result.getDouble(1)>=0.5) {
						result = calibrator.output(input);
						double sum = 0.0;
						for (int k=0; k<45; k++) {
							if (result.getDouble(k)>=0.1) {
								sum += result.getDouble(k);
							}
						}
						double sn=0.0, xn=0.0, yn=0.0;
						for (int k=0; k<45; k++) {
							if (result.getDouble(k)>=0.1) {
								sn = sn + result.getDouble(k)/sum * scales[k][0];
								xn = xn + result.getDouble(k)/sum * scales[k][1];
								yn = yn + result.getDouble(k)/sum * scales[k][2];
							}
						}
						newX=j*stride+(int)(xn*kernelSize);
						newY=i*stride+(int)(yn*kernelSize);
						newW=(int)(Math.pow(1.1, sn)*kernelSize);
						if (newX<0 || newY<0 ||
								newX+newW>image.cols() || newY+newW>image.rows()) {
							frames.add(new Frame(j*stride, i*stride, kernelSize));
						} else {
							frames.add(new Frame(newX, newY, newW));
						}
					}
				} catch (IOException e) {
					log.error("Error converting from Mat to INDArray!");
					e.printStackTrace();
				}

				frame.release();
			}
		}
		
		
		List<Frame> filteredFrames = new ArrayList<Frame>();
		List<Frame> group = new ArrayList<Frame>();

		double minIoU = 0.5;
		
		while (frames.size()>0) {
			if (frames.size()==1) {
				filteredFrames.add(frames.get(0));
				break;
			}
			group.clear();
			group.add(frames.get(0));
			for (int i=1; i<frames.size(); i++) {
				if (iOu(group.get(0), frames.get(i))>minIoU) {
					group.add(frames.get(i));
				}
			}
			frames.removeAll(group);
			filteredFrames.add(getBestFrame(image, group));
		}

		return filteredFrames;
	}
	
	
	private static Frame getBestFrame(Mat image, List<Frame> group) {
		
		Frame bestFrame = null;
		double bestScore = -1.0;
		Mat frame;
		for (Frame f:group) {
			frame = image.colRange(f.x, f.x+f.w).rowRange(f.y, f.y+f.w);
			try {
				INDArray input = new NativeImageLoader(13, 13, image.channels()).asMatrix(frame);
				ipps.preProcess(input);
				INDArray result = detector.output(input);
				if (bestScore<result.getDouble(1)) {
					bestFrame = f;
					bestScore = result.getDouble(1);
				}
			} catch (IOException e) {
				log.error("Error converting from Mat to INDArray!");
				e.printStackTrace();
			}
			frame.release();
		}
		
		return bestFrame;
	}
	
	private static double iOu(Frame f1, Frame f2) {
		
		int intersanction = 0;
		int union = 0;
		
		intersanction =
				(Math.min(f1.x+f1.w, f2.x+f2.w) -
				 Math.max(f1.x,      f2.x)) *
				(Math.min(f1.y+f1.w, f2.y+f2.w) -
				 Math.max(f1.y,      f2.y));
		
		intersanction = Math.max(0, intersanction);
		
		union = f1.w * f1.w + f2.w * f2.w - intersanction;
		
		return (double)intersanction/union;
	}
	
	private static void saveFrames(Mat image, List<Frame> frames, String suffix) {

		Mat frame = null;
		for (Frame f:frames) {
			if (f.x<0 || f.y<0 || f.x+f.w>image.cols() || f.y+f.w>image.rows()) {
				log.warn("Frame is not within the image!");
				continue;
			}
			frame = image.colRange(f.x, f.x+f.w).rowRange(f.y, f.y+f.w);
			imwrite("/home/ambra/Deep Learning/ImageNet/detected/"+saveCounter+
					"_" + suffix + ".png", frame);
			saveCounter++;
			frame.release();
		}
		
	}
	
}
