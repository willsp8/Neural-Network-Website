package com.example.neural_network_backend.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;


@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Document(collection="message")
public class MessagesDTO {

    @Id
    private String id; 
    private String senderName;
    private String receiverName;
    private String message; 
    private String chatRoomId; 
    private String date;
    private Boolean seen;
    private Status status;
    
}

