package com.ambraspace.recognition.opencv;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.INTER_CUBIC;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.SimpleLayout;
import org.apache.log4j.spi.Filter;
import org.apache.log4j.spi.LoggingEvent;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ImageResize {
	
    private static final Logger log = LoggerFactory.getLogger(ImageResize.class);
    
    
    public static class ToResize {
    	
    	private File srcImage;
    	private File dstImage;
    	private int size;
    	
    	public ToResize(File srcImage, File dstImage, int size) {
    		this.srcImage = srcImage;
    		this.dstImage = dstImage;
    		this.size = size;
		}

		public File getSrcImage() {
			return srcImage;
		}

		public File getDstImage() {
			return dstImage;
		}

		public int getSize() {
			return size;
		}
    	
    }
    
    
    public static class ResizeContainer {
    	
    	private List<ToResize> container = new ArrayList<ToResize>();
    	
    	public synchronized void insert(ToResize tr) {
    		container.add(tr);
    	}
    	
    	public synchronized ToResize pullNext() {
    		
    		if (container.size()==0) {
    			return null;
    		} else {
    			ToResize tr = container.get(0);
    			container.remove(0);
    			return tr;
    		}
    		
    	}
    	
    }
    
    
    public static class Resizer implements Runnable {
    	
    	private ResizeContainer container;
    	
    	public Resizer(ResizeContainer container) {

    		this.container = container;

		}

		@Override
		public void run() {
			
			ToResize tr = null;
	    	
			Mat srcMat = null;
	    	Mat dstMat = new Mat();
		
	    	while ((tr=container.pullNext())!=null) {
				srcMat = imread(tr.getSrcImage().getAbsolutePath());
				resize(
						srcMat,
						dstMat,
						new Size(tr.getSize(), tr.getSize()),
						1.0d, 1.0d,
						INTER_CUBIC);
				imwrite(tr.getDstImage().getAbsolutePath(), dstMat);
				srcMat.release();
			}

		}
    	
    	
    }
	
	public static void main(String[] args) {
		
		ConsoleAppender ca = new ConsoleAppender(new SimpleLayout());
		ca.addFilter(new Filter() {
			@Override
			public int decide(LoggingEvent event) {
				if (event.getLevel().equals(Level.DEBUG)) {
					return Filter.DENY;
				} else {
					return Filter.ACCEPT;
				}
			}
		});
		BasicConfigurator.configure(ca);

		int iSize=51;
		File srcDir = new File("/home/ambra/Deep Learning/LFW/calibration");
		File dstDir = new File("/home/ambra/Deep Learning/LFW/calibration-51");
		
		ExecutorService threadPool = Executors.newFixedThreadPool(8);
		ResizeContainer rc = new ResizeContainer();
		
		File currentDir = null;
		String dirName = null;
		File[] files = null;
		for (int i=0; i<45; i++) {
			
			dirName = String.format("%02d", i);
			currentDir = new File(srcDir, dirName);
			new File(dstDir, dirName).mkdir(); 
	
			files = currentDir.listFiles();
			ToResize tr = null;
			for (int j=0; j<files.length; j++) {
				tr = new ToResize(
						files[j],
						new File(dstDir.getAbsolutePath() + File.separatorChar +
								dirName + File.separatorChar + files[j].getName()),
						iSize);
				rc.insert(tr);
			}	
		}
		
		for (int i=0; i<8; i++) {
			threadPool.submit(new Resizer(rc));
		}
		
		threadPool.shutdown();
		log.info("Threads started.");
		
	}

}
