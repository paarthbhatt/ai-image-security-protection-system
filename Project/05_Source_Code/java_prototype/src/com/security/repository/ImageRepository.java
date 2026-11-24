package com.security.repository;

import com.security.model.ImageRecord;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Thread-safe repository managing image assets in memory.
 */
public class ImageRepository {
    private final List<ImageRecord> records = new ArrayList<>();

    public synchronized void add(ImageRecord record) {
        records.add(record);
    }

    public synchronized List<ImageRecord> findAll() {
        return new ArrayList<>(records);
    }

    public synchronized Optional<ImageRecord> findById(int id) {
        return records.stream().filter(r -> r.getId() == id).findFirst();
    }

    public synchronized Optional<ImageRecord> findByHash(String sha256) {
        return records.stream().filter(r -> r.getSha256Digest().equalsIgnoreCase(sha256)).findFirst();
    }

    public synchronized boolean delete(int id) {
        return records.removeIf(r -> r.getId() == id);
    }

    public synchronized int count() {
        return records.size();
    }
}
