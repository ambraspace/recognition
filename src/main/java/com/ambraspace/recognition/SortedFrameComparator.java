package com.ambraspace.recognition;

import java.util.Comparator;

public class SortedFrameComparator implements Comparator<SortedFrame> {

	@Override
	public int compare(SortedFrame o1, SortedFrame o2) {
		if (o1.getScore()>o2.getScore()) {
			return -1;
		}
		if (o1.getScore()<o2.getScore()) {
			return 1;
		}
		if (o1.hashCode()>o2.hashCode()) {
			return 1;
		}
		if (o1.hashCode()<o2.hashCode()) {
			return -1;
		}
		return 0;
	}

}
