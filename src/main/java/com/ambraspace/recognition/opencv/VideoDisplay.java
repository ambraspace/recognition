package com.ambraspace.recognition.opencv;

import static org.bytedeco.javacpp.opencv_highgui.CV_WINDOW_AUTOSIZE;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.namedWindow;
import static org.bytedeco.javacpp.opencv_highgui.startWindowThread;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RGBA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_videoio.CAP_V4L2;

import java.io.IOException;
import java.util.List;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ambraspace.recognition.Frame;
import com.ambraspace.recognition.FullCascade1;

public class VideoDisplay {
	
	private static final Logger log = LoggerFactory.getLogger(VideoDisplay.class);
	
	public static void main(String[] args) throws IOException {
		
		
		MultiLayerNetwork net13detection = null;
		MultiLayerNetwork net25detection = null;
		MultiLayerNetwork net51detection = null;
		MultiLayerNetwork net13calibration = null;
		MultiLayerNetwork net25calibration = null;
		MultiLayerNetwork net51calibration = null;

//		try {
//			
//			net13detection =
//					ModelSerializer.restoreMultiLayerNetwork(
//							"/home/ambra/Deep Learning/LFW/detection-13/9906.zip", false);
//			net25detection =
//					ModelSerializer.restoreMultiLayerNetwork(
//							"/home/ambra/Deep Learning/LFW/detection-25/9968.zip", false);
//			net51detection =
//					ModelSerializer.restoreMultiLayerNetwork(
//							"/home/ambra/Deep Learning/LFW/detection-51/9937.zip", false);
//			net13calibration =
//					ModelSerializer.restoreMultiLayerNetwork(
//							"/home/ambra/Deep Learning/LFW/calibration-13/6942.zip", false);
//			net25calibration =
//					ModelSerializer.restoreMultiLayerNetwork(
//							"/home/ambra/Deep Learning/LFW/calibration-25/7963.zip", false);
//			net51calibration =
//					ModelSerializer.restoreMultiLayerNetwork(
//							"/home/ambra/Deep Learning/LFW/calibration-51/9112.zip", false);
//			
//		} catch (IOException err) {
//			log.error("Unable to initialize the networks!");
//			err.printStackTrace();
//			System.exit(1);
//		}
//		
//		FullCascade1 fc1 = new FullCascade1(
//				net13detection,
//				net13calibration,
//				net25detection,
//				net25calibration,
//				net51detection,
//				net51calibration);

		
		VideoCapture video = new VideoCapture();
		boolean OK = video.open(0+CAP_V4L2);
		OK = video.grab();
		Mat image = new Mat();
		Mat gray = new Mat();
		namedWindow("video", CV_WINDOW_AUTOSIZE);
		startWindowThread();
		while (true) {
			video.read(image);
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
	                throw new IOException("Cannot convert from " + image.channels()
	                                                    + " to " + 1 + " channel.");
	            }
	            cvtColor(image, gray, code);
	        }
	        
//	        long start, stop;
//	        
//	        start = System.currentTimeMillis();
//			List<Frame> detectedFrames = fc1.detectFaces(gray, 40, Integer.MAX_VALUE);
//			stop = System.currentTimeMillis();
//			log.info("It took " + (stop-start) + " ms.");
//			
//			if (detectedFrames.size()>0) {
//				for (Frame f:detectedFrames) {
//					rectangle(image, new Rect(f.x, f.y, f.w, f.w), new Scalar(0.0, 255.0, 0.0, 0.0));
//				}
//			}
	        
			imshow("video", image);
			
//			image.release();
//			gray.release();
		}

	}

}
