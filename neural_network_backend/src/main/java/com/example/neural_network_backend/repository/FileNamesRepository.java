package com.example.neural_network_backend.repository;

import java.util.List;
import java.util.Optional;

import org.springframework.data.mongodb.repository.MongoRepository;

import com.example.neural_network_backend.model.FileNamesDTO;

public interface FileNamesRepository extends MongoRepository<FileNamesDTO, String> {
    List<FileNamesDTO> findByUserId(String userId);
}
