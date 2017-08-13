package com.ambraspace.recognition.dl4j;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class MyCNN {
	
    private static final Logger log = LoggerFactory.getLogger(MyCNN.class);
	
	public static void main(String[] args) {
		
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);

		
		double bestRecall = 0.5f;
		
		int imageWidth = 51;
		int imageHeight = 51;
		int channels = 1;
		int outputNum = 2;
		int numEpochs = 100;
		int batchSize = 50;
		int rngSeed = 123;
		Random randNumGen = new Random(rngSeed);
		int iterations = 1;

		
        File parentDir = new File("/home/ambra/Deep Learning/LFW/training/classification1/");
        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, new String[]{"png"}, randNumGen);
        
        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

//        BalancedPathFilter pathFilter = new BalancedPathFilter(
//        		randNumGen,
//        		new String[]{"png"},
//        		labelMaker, 3000);
        RandomPathFilter pathFilter = new RandomPathFilter(randNumGen, new String[]{"png"});
        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];
        
        ImageRecordReader rr1 = new ImageRecordReader(imageHeight, imageWidth, channels, labelMaker);
        ImageRecordReader rr2 = new ImageRecordReader(imageHeight, imageWidth, channels, labelMaker);

        try {
			rr1.initialize(trainData);
	        rr2.initialize(testData);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
        
        DataSetIterator trainDataIterator = new RecordReaderDataSetIterator(rr1, batchSize, -1, outputNum);
        DataSetIterator testDataIterator = new RecordReaderDataSetIterator(rr2, batchSize, -1, outputNum);
        
        trainDataIterator.setPreProcessor(new ImagePreProcessingScaler());
        testDataIterator.setPreProcessor(new ImagePreProcessingScaler());

        MultiLayerNetwork model = null;
        
        File modelFile = new File("model.zip");
        if (modelFile.exists()) {
        	log.info("Loading prepared model.");
        	try {
				model = ModelSerializer.restoreMultiLayerNetwork(modelFile, true);
			} catch (IOException e) {
				log.error("Unable to load the model!");
			}
        }
		
		
        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .iterations(iterations)
                .regularization(true).l1(0.002)
                .learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.6)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(128)
                        .activation("identity")
                        .build())
                .layer(1, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .stride(1, 1)
                        .nOut(64)
                        .activation("identity")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .stride(1, 1)
                        .nOut(32)
                        .activation("identity")
                        .build())
                .layer(3, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .stride(1, 1)
                        .nOut(16)
                        .activation("identity")
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(640).build())
                .layer(5, new DenseLayer.Builder().activation("relu")
                        .nOut(64).build())
                .layer(6, new OutputLayer.Builder(
                		new LossBinaryXENT(new NDArray(new double[][]{{1.8, 0.5}})))
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        builder.setInputType(InputType.convolutional(imageHeight, imageWidth, channels));
        
        MultiLayerConfiguration conf = builder.build();
        
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        if (model!=null) { 
        	network.setParams(model.params());
        }
        
        log.info("Train model...");
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
//        network.setListeners(new ScoreIterationListener(1));
        double edge_d = 0.0;
        double edge = 0.5;
        double edge_u = 1.0;
        for (int i=0; i<numEpochs+0; i++ ) {
        	Long start = System.currentTimeMillis();
        	network.fit(trainDataIterator);
            Long stop = System.currentTimeMillis();
            log.info("*** Completed epoch {} ***", i);
            log.info("It took " + (stop-start)/1000 + " seconds.");
            log.info("Evaluate model...");
            Evaluation eval = new Evaluation(outputNum);
            long tp=0, tn=0, fp=0, fn=0;
            while(testDataIterator.hasNext()){
                DataSet ds = testDataIterator.next();
                INDArray input = ds.getLabels();
                INDArray output = network.output(ds.getFeatureMatrix(), false);
//                for (int j=0; j<output.rows(); j++) {
//                	if (output.getDouble(j, 1)>=edge) {
//                		if (input.getDouble(j, 1)>=0.5) {
//                			tp++;
//                		} else {
//                			fp++;
//                		}
//                	} else {
//                		if (input.getDouble(j, 0)>=0.5) {
//                			tn++;
//                		} else {
//                			fn++;
//                		}
//                	}
//                }
                eval.eval(ds.getLabels(), output);
            }
//            double fErrorRate=(double)fp/tn*100;
//            double mErrorRate=(double)fn/tp*100;
//            System.out.printf("TN: %d, FP: %d, FN: %d, TP: %d\n", tn, fp, fn, tp);
//            System.out.printf("F error rate: %.2f, M error rate: %.2f\n", fErrorRate, mErrorRate);
//            if (mErrorRate>fErrorRate) {
//            	edge_u=edge;
//            	edge=(edge_u+edge_d)/2.0;
//            } else if (mErrorRate<fErrorRate) {
//            	edge_d=edge;
//            	edge=(edge_u+edge_d)/2.0;
//            }
//            System.out.printf("Setting edge to: %.4f\n", edge);
            log.info(eval.stats());
            testDataIterator.reset();
            
//            if (eval.recall()>bestRecall) {
            	bestRecall=eval.recall();
                try {
                	ModelSerializer.writeModel(network, String.format("%05d-%4.4f", i, bestRecall)+".zip", true);
    			} catch (IOException e) {
    				log.error("Can not save model!");
    			}
//            }

        }
        log.info("****************Example finished********************");

	}
	
}
