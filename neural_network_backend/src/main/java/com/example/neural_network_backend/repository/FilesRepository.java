package com.example.neural_network_backend.repository;

import org.springframework.data.mongodb.repository.MongoRepository;

import com.example.neural_network_backend.model.FilesDTO;


public interface FilesRepository extends MongoRepository<FilesDTO, String>{

    
}
