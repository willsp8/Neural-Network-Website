package com.example.neural_network_backend.repository;

import java.util.Optional;

import org.springframework.data.mongodb.repository.MongoRepository;

import com.example.neural_network_backend.model.Chat;

import java.util.List;


public interface ChatRepository extends MongoRepository<Chat, String> {
    Optional<Chat> findByRoom(String room);
    
    
} 
