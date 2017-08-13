package com.ambraspace.recognition;

public class SortedFrame extends Frame {
	
	private double score = 0.0;

	public SortedFrame(int x, int y, int w) {
		super(x, y, w);
	}
	
	public SortedFrame(Frame f) {
		super(f.x, f.y, f.w);
	}
	
	public SortedFrame(int x, int y, int w, double score) {
		super(x, y, w);
		this.score = score;
	}
	
	public SortedFrame(SortedFrame originalFrame) {
		super(originalFrame.x, originalFrame.y, originalFrame.w);
		this.score = originalFrame.getScore();
	}

	public double getScore() {
		return score;
	}

	public void setScore(double score) {
		this.score = score;
	}

	@Override
	public String toString() {
		return String.format("SortedFrame (x=%d, y=%d, w=%d, score=%.4f)", x, y, w, score);
	}
	
}
