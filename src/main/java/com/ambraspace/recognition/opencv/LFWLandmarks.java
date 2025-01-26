package com.ambraspace.recognition.opencv;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.INTER_AREA;
import static org.bytedeco.javacpp.opencv_imgproc.line;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.SimpleLayout;
import org.apache.log4j.spi.Filter;
import org.apache.log4j.spi.LoggingEvent;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LFWLandmarks {
	
	private static final Logger log = LoggerFactory.getLogger(LFWLandmarks.class);
	

	private static class Image {
		
		private File srcFile;
		private List<Face> faces;

		public Image(File src) {
			this.srcFile = src;
			this.faces = new ArrayList<Face>();
		}
		
		public File getSrcFile() {
			return srcFile;
		}
		
		public List<Face> getFaces() {
			return faces;
		}
		
		public void addFace(Face f) {
			if (f==null)
				return;
			faces.add(f);
		}
		
	}
	
	
	private static class Face {
		
		private List<Point> points;
		
		public Face() {
			this.points = new ArrayList<Point>();
		}
		
		public List<Point> getPoints() {
			return points;
		}
		
		public void addPoint(Point p) {
			
			if (p==null)
				return;
			
			points.add(p);
		}
		
	}
	
	
	private static boolean frameIsValid(int x1, int x2, int y1, int y2, int imW, int imH) {
		
		double w, h, cx, cy;
		w=x2-x1+1;
		h=y2-y1+1;
		cx=x1+w/2;
		cy=y1+h/2;
		w=Math.max(w, h)*Math.pow(1.1, 4);
		if (Math.round(cx-w/2.0-1.0/6.0*w)<0) return false;
		if (Math.round(cy-w/2.0-1.0/6.0*w)<0) return false;
		if (Math.round(cx+w/2.0+1.0/6.0*w)>=imW) return false;
		if (Math.round(cy+w/2.0+1.0/6.0*w)>=imH) return false;
		return true;
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
		
		/*
		 * Collect all landmarks from TXT file.
		 */
		
		File inputFile = new File("/home/ambra/Deep Learning/LFW/lfw_landmarks.txt");
		List<Image> allLm = new ArrayList<Image>();

		try (BufferedReader fin = new BufferedReader(
				new InputStreamReader(
						new FileInputStream(inputFile)))) {
			
			String line = null;
			int numOfFaces, numOfParts;
			Image currentImage = null;
			Face currentFace = null;
			int x, y;
			while ((line=fin.readLine())!=null) {
				if (line.startsWith("processing image")) {
					line = line.substring(17);
					currentImage = new Image(new File(line));
					allLm.add(currentImage);
						
					line = fin.readLine();
					numOfFaces = Integer.parseInt(line.substring(26));
					for (int i=0; i<numOfFaces; i++) {
						currentFace = new Face();
						currentImage.addFace(currentFace);
						line=fin.readLine();
						numOfParts = Integer.parseInt(line.substring(17));
						for (int j=0; j<numOfParts; j++) {
							line=fin.readLine();
							line = line.substring(29);
							x=Integer.parseInt(line.substring(1, line.indexOf(",")));
							y=Integer.parseInt(line.substring(line.indexOf(",")+2, line.indexOf(")")));
							currentFace.addPoint(new Point(x, y));
						}
					}
				}
			}
		} catch (FileNotFoundException err) {
			log.error("File " + inputFile.getAbsolutePath() + " not found!");
		} catch (IOException err) {
			log.error("Unable to read " + inputFile.getAbsolutePath());
		}

			
		{ // Print all landmarks to console.
			if (false) {
				Iterator<Image> it = allLm.iterator();
				Image image = null;
				List<Face> faces = null;
				List<Point> points = null;
				while (it.hasNext()) {
					image = it.next();
					faces = image.getFaces();
					for (int i=0; i<faces.size(); i++) {
						System.out.println(image.getSrcFile().getAbsolutePath() + "-" + i);
						points = faces.get(i).getPoints();
						for (int j=0; j<points.size(); j++) {
							System.out.println(points.get(j).x() + ", " + points.get(j).y());
						}
					}
				}
			}
		}
		
		/*
		 * Overlay landmarks on images, and save images to output folder
		 * for manual selection of properly recognized landmarks.
		 */
		{
		
			if (false) {
				Iterator<Image> it = allLm.iterator();
				Image image = null;
				List<Face> faces = null;
				List<Point> points = null;
				Mat srcImage = null;
				int x1, y1, x2, y2, w, h;
				while (it.hasNext()) {
					image = it.next();
					srcImage = imread(image.getSrcFile().getAbsolutePath());
					faces = image.getFaces();
					for (int i=0; i<faces.size(); i++) {
						
						Mat dstImage = new Mat();
						
						points = faces.get(i).getPoints();

						x1 = x2 = points.get(17).x();
						y1 = y2 = points.get(17).y();
						for (int j=17; j<points.size(); j++) {
							if (x1>points.get(j).x()) x1=points.get(j).x(); 
							if (x2<points.get(j).x()) x2=points.get(j).x();
							if (y1>points.get(j).y()) y1=points.get(j).y(); 
							if (y2<points.get(j).y()) y2=points.get(j).y();
						}
						
						if (!frameIsValid(x1, x2, y1, y2, srcImage.arrayWidth(), srcImage.arrayHeight()))
							continue;

						w=x2-x1+1;
						h=y2-y1+1;
						if (w<h) {
							x1=x1-(h-w)/2;
							x2=x1+h-1;
							w=h;
						} else if (h<w) {
							y1=y1-(w-h)/2;
							y2=y1+w-1;
							h=w;
						}
						x1=x1-(int)(w*0.2);
						x2=x2+(int)(w*0.2);
						y1=y1-(int)(w*0.2);
						y2=y2+(int)(w*0.2);
						w=x2-x1+1;
						h=y2-y1+1;
						
						for (int j=17; j<21; j++) {
							line(srcImage, points.get(j), points.get(j+1), new Scalar(0, 255, 0, 0));
						}
						for (int j=22; j<26; j++) {
							line(srcImage, points.get(j), points.get(j+1), new Scalar(0, 255, 0, 0));
						}
						for (int j=27; j<35; j++) {
							line(srcImage, points.get(j), points.get(j+1), new Scalar(0, 255, 0, 0));
						}
						line(srcImage, points.get(35), points.get(30), new Scalar(0, 255, 0, 0));
						for (int j=36; j<41; j++) {
							line(srcImage, points.get(j), points.get(j+1), new Scalar(0, 255, 0, 0));
						}
						line(srcImage, points.get(41), points.get(36), new Scalar(0, 255, 0, 0));
						for (int j=42; j<47; j++) {
							line(srcImage, points.get(j), points.get(j+1), new Scalar(0, 255, 0, 0));
						}
						line(srcImage, points.get(47), points.get(42), new Scalar(0, 255, 0, 0));
						for (int j=48; j<59; j++) {
							line(srcImage, points.get(j), points.get(j+1), new Scalar(0, 255, 0, 0));
						}
						line(srcImage, points.get(59), points.get(48), new Scalar(0, 255, 0, 0));
						for (int j=60; j<67; j++) {
							line(srcImage, points.get(j), points.get(j+1), new Scalar(0, 255, 0, 0));
						}
						line(srcImage, points.get(67), points.get(60), new Scalar(0, 255, 0, 0));

						dstImage = srcImage.colRange(x1, x2+1).rowRange(y1, y2+1);

						resize(dstImage, dstImage, new Size(200, 200));
						imwrite("/home/ambra/Deep Learning/LFW/WITH LANDMARKS/" +
						image.getSrcFile().getName().substring(0, image.getSrcFile().getName().lastIndexOf("."))+"-"+i+".png", dstImage);
						dstImage.release();
					}
					srcImage.release();
				}
			}
		}
		
		/*
		 * Generation of all necessary classes for CNN training.
		 */
		{
			if (true) {
				
				File calibration13folder = new File("/home/ambra/Deep Learning/LFW/calibration-13/");
				File calibration25folder = new File("/home/ambra/Deep Learning/LFW/calibration-25/");
				File calibration51folder = new File("/home/ambra/Deep Learning/LFW/calibration-51/");
				File calibrationFolder = new File("/home/ambra/Deep Learning/LFW/calibration/");
				
				if (!calibration13folder.exists()) calibration13folder.mkdirs();
				if (!calibration25folder.exists()) calibration25folder.mkdirs();
				if (!calibration51folder.exists()) calibration51folder.mkdirs();
				if (!calibrationFolder.exists()) calibrationFolder.mkdirs();
				
				Iterator<Image> it = allLm.iterator();
				Image image = null;
				List<Face> faces = null;
				Mat srcImage = null;
				Mat dstImage = new Mat();
				Mat frameImage = new Mat();
				int xmin, xmax, ymin, ymax;
				double w0, w, h, cx, cy, xres, yres;

				while (it.hasNext()) {
					image = it.next();
					srcImage = imread(image.getSrcFile().getAbsolutePath());
					faces = image.getFaces();
					File dstFile = null;
					List<Point> points = null;
					for (int i=0; i<faces.size(); i++) {
						dstFile = new File("/home/ambra/Deep Learning/LFW/WITH LANDMARKS/" +
								image.getSrcFile().getName()
								.substring(0, image.getSrcFile().getName().lastIndexOf(".")) + "-" +
										i + ".png");
						if (!dstFile.exists()) {
							continue;
						}
						
						points = faces.get(i).getPoints();
						xmin = xmax = points.get(17).x();
						ymin = ymax = points.get(17).y();
						
						for (int j=17; j<points.size(); j++) {
							if (xmin>points.get(j).x()) xmin=points.get(j).x(); 
							if (xmax<points.get(j).x()) xmax=points.get(j).x();
							if (ymin>points.get(j).y()) ymin=points.get(j).y(); 
							if (ymax<points.get(j).y()) ymax=points.get(j).y();
						}
						
						w=xmax-xmin+1;
						h=ymax-ymin+1;
						cx=xmin+w/2.0;
						cy=ymin+h/2.0;
						
						w0=Math.max(w, h);
						
						String subFolderName = null;
						File subFolder = null;
						for (int k=4; k>=0; k--) {
							w=w0*Math.pow(1.1, k);
							for (int l=-1; l<=1; l++) {
								for (int m=-1; m<=1; m++) {

									subFolderName = String.format("%02d",
											(4-k)*9+(l+1)*3+(m+1));
									
									subFolder = new File(calibration13folder, subFolderName);
									if (!subFolder.exists()) subFolder.mkdirs();
									subFolder = new File(calibration25folder, subFolderName);
									if (!subFolder.exists()) subFolder.mkdirs();
									subFolder = new File(calibration51folder, subFolderName);
									if (!subFolder.exists()) subFolder.mkdirs();
									subFolder = new File(calibrationFolder, subFolderName);
									if (!subFolder.exists()) subFolder.mkdirs();

									xres=cx-w/2.0-1.0/6.0*l*w;
									yres=cy-w/2.0-1.0/6.0*m*w;
									
									dstImage = srcImage.colRange(
											(int)Math.round(xres),
											(int)Math.round(xres+w)+1)
											.rowRange(
											(int)Math.round(yres),
											(int)Math.round(yres+w)+1);
									
//									resize(dstImage, frameImage, new Size(13, 13), 1.0, 1.0, INTER_AREA);
//									imwrite(calibration13folder.getAbsolutePath() +
//											File.separator + subFolderName + File.separator +
//											dstFile.getName(),
//											frameImage);
//									frameImage.release();
//									resize(dstImage, frameImage, new Size(25, 25), 1.0, 1.0, INTER_AREA);
//									imwrite(calibration25folder.getAbsolutePath() +
//											File.separator + subFolderName + File.separator +
//											dstFile.getName(),
//											frameImage);
//									frameImage.release();
//									resize(dstImage, frameImage, new Size(51, 51), 1.0, 1.0, INTER_AREA);
//									imwrite(calibration51folder.getAbsolutePath() +
//											File.separator + subFolderName + File.separator +
//											dstFile.getName(),
//											frameImage);
//									frameImage.release();

									imwrite(calibrationFolder.getAbsolutePath() +
									File.separator + subFolderName + File.separator +
									dstFile.getName(),
									dstImage);

									dstImage.release();
								}
							}
						}
						
					}
					srcImage.release();
				}
				
			}
		}
			
			
		/*
		 * Resize images for training speed-up
		 */
		{
			if (true) {
				
				int size = 51;
				int channels = org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_GRAYSCALE;
				
				File srcDir = new File("/home/ambra/Deep Learning/LFW/calibration/");
				File dstDir = new File("/home/ambra/Deep Learning/LFW/calibration-" + size + "/");
				
				Mat srcImage = new Mat();
				Mat dstImage = new Mat();
				
				File[] files = null;
				for (int dir=0; dir<45; dir++) {
					
					files = new File(srcDir, String.format("%02d", dir)).listFiles();
					for (File f:files) {
						
						srcImage = imread(f.getAbsolutePath(), channels);
						resize(srcImage, dstImage, new Size(size, size));
						imwrite(dstDir.getAbsolutePath()+File.separator +
								String.format("%02d", dir) + 
								File.separator + f.getName(), dstImage);
						
						srcImage.release();
						dstImage.release();
						
					}
					
					
				}
				
			}
		}
				
		
	}

}
