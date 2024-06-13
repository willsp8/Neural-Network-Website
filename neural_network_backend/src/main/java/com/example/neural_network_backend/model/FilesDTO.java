package com.example.neural_network_backend.model;

import java.util.ArrayList;
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
@Document(collection="file")
public class FilesDTO {

    @Id
    private String id;

    // we need a byte array that will be our csv file 
    private byte[] csvFile;

    // we need our photos to save to the data base 
    private List<byte[]> photos;
    
}
