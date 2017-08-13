package com.ambraspace.recognition.dl4j;

import java.util.Collection;
import java.util.Iterator;

import com.flickr4java.flickr.Flickr;
import com.flickr4java.flickr.FlickrException;
import com.flickr4java.flickr.REST;
import com.flickr4java.flickr.Transport;
import com.flickr4java.flickr.machinetags.Namespace;
import com.flickr4java.flickr.machinetags.NamespacesList;
import com.flickr4java.flickr.photos.Photo;
import com.flickr4java.flickr.photos.PhotoList;
import com.flickr4java.flickr.photos.SearchParameters;
import com.flickr4java.flickr.tags.HotlistTag;
import com.flickr4java.flickr.tags.RelatedTagsList;
import com.flickr4java.flickr.tags.Tag;

public class Test {

	public static void main	(String[] args) {
		
		com.flickr4java.flickr.Flickr f = new Flickr("87b23f815ff95a7db13ef519639785d5", "5f72d71f335a80e1", new REST());
		try {
			SearchParameters sp = new SearchParameters();
			sp.setTags(new String[]{"mugshot"});
			PhotoList<Photo> pl = f.getPhotosInterface().search(sp, 100, 0);
			Iterator<Photo> it = pl.iterator();
			while (it.hasNext()) {
				System.out.println(it.next().getSmall320Url());
			}
		} catch (FlickrException e1) {
			e1.printStackTrace();
		}
	}
}
