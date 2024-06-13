package com.example.neural_network_backend.repository;

import org.springframework.data.mongodb.repository.MongoRepository;

import com.example.neural_network_backend.model.UserDTO;

import java.util.Optional;


public interface UserRepository extends MongoRepository<UserDTO, String>  {
    Optional<UserDTO> findByEmail(String email);
}
