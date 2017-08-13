package com.ambraspace.recognition.dl4j;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Net51calibration {

	
    private static final Logger log = LoggerFactory.getLogger(Net51calibration.class);
	
	public static void main(String[] args) {
		
//		UIServer uiServer = UIServer.getInstance();
//		StatsStorage statsStorage = new InMemoryStatsStorage();
//		uiServer.attach(statsStorage);

		
		int imageWidth = 51;
		int imageHeight = 51;
		int channels = 1;
		int outputNum = 45;
		int numEpochs = 1000;
		int batchSize = 180;
		int rngSeed = 123;
		Random randNumGen = new Random(rngSeed);
		int iterations = 1;
		double bestRecall = 1.0/outputNum;
		

		
        File parentDir = new File("/home/ambra/Deep Learning/LFW/calibration-51/");
        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, new String[]{"png"}, randNumGen);
        
        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        BalancedPathFilter pathFilter = new BalancedPathFilter(
        		randNumGen,
        		new String[]{"png"},
        		labelMaker);
        //RandomPathFilter pathFilter = new RandomPathFilter(randNumGen, new String[]{"jpg"});
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
        
        ImagePreProcessingScaler ipps = new ImagePreProcessingScaler();
        trainDataIterator.setPreProcessor(ipps);
        testDataIterator.setPreProcessor(ipps);
        
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
                .regularization(false)
                .learningRate(0.001)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        //.activation("relu")
                        .build())
                .layer(2, new LocalResponseNormalization.Builder()
                		//.alpha(0)
                		//.beta(0)
                		//.k(0)
                		.n(9)
                		//.activation("relu")
                		.build())
                .layer(3, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new LocalResponseNormalization.Builder()
                		//.alpha(0)
                		//.beta(0)
                		//.k(0)
                		.n(9)
                		//.activation("relu")
                		.build())
                .layer(5, new DenseLayer.Builder()
                		.activation(Activation.RELU)
                        .nOut(256)
                        .build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
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
//        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
        network.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<numEpochs; i++ ) {
        	Long start = System.currentTimeMillis();
        	network.fit(trainDataIterator);
            Long stop = System.currentTimeMillis();
            log.info("*** Completed epoch {} ***", i);
            log.info("It took " + (stop-start)/1000 + " seconds.");
            log.info("Evaluate model...");
            Evaluation eval = new Evaluation(outputNum);
            while(testDataIterator.hasNext()){
                DataSet ds = testDataIterator.next();
                INDArray output = network.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }
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
