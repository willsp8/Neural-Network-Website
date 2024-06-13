package com.example.neural_network_backend.model;

import java.util.List;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.DBRef;


import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.mongodb.core.mapping.Document;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "Users")
public class UserDTO {
    

    @Id
    private String id; 

    private String name; 

    private String email;

    private UserRole role;

    private RegistrationSource source;
    
    private String photoId;

    private String ProfileName;

    private String lastMessage;

    private boolean seenMesssage;

    ByteArrayResource resourceImage;

    @DBRef
    private List<FileNamesDTO> fileNames;


}

