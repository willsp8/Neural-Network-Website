package com.example.neural_network_backend.model;

import java.util.List;

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
@Document(collection="chat")
public class Chat {
    @Id
    private String roomId;

    private String room;

    private List<UserDTO> users; 

    private List<MessagesDTO> messages;
    
}
