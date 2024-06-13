package com.example.neural_network_backend.repository;

import java.util.List;

import org.springframework.data.mongodb.repository.MongoRepository;

import com.example.neural_network_backend.model.MessagesDTO;


public interface MessageRepository extends MongoRepository<MessagesDTO, String>{
    List<MessagesDTO> findByReceiverName(String receiverName);
    List<MessagesDTO> findByChatRoomId(String chatRoomId);
    
}
