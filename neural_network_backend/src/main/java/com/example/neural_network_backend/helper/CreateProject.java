package com.example.neural_network_backend.helper;

import java.net.http.HttpResponse;
import java.util.List;

import org.springframework.data.annotation.Id;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class CreateProject {

    private Object columns;

    
    private String id; 
    
}
