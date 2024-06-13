package com.example.neural_network_backend.controller;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.Principal;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Optional;

import javax.imageio.ImageIO;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.data.repository.query.Param;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PostAuthorize;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Controller;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import com.example.neural_network_backend.helper.ChatHelper;
import com.example.neural_network_backend.model.Chat;
import com.example.neural_network_backend.model.MessagesDTO;
import com.example.neural_network_backend.model.UserDTO;
import com.example.neural_network_backend.repository.ChatRepository;
import com.example.neural_network_backend.repository.FilesRepository;
import com.example.neural_network_backend.repository.MessageRepository;
import com.example.neural_network_backend.repository.UserRepository;
import com.example.neural_network_backend.sevice.AwsService;

import jakarta.servlet.http.HttpServletRequest;

@RestController
@RequestMapping("/")
@CrossOrigin("http://localhost:5173")
@Controller
public class NeuralNeworkController {
    
    @Autowired
    private UserRepository userRepo;

    @Autowired
    private FilesRepository filesRepo;

    @Autowired 
    private AwsService awsService;

    @Autowired 
    private MessageRepository messageRepo;

    @Autowired
    private ChatRepository chatRepo;

    
    @GetMapping("/test")
    public String getMethodUsers2() {
        return "userRepo.findAll();";
    }


    @GetMapping("/testing")
    public ResponseEntity<String> getTesting() {
        return ResponseEntity.ok("Wokring");
    }


  
    // @GetMapping("/users")
    // public ResponseEntity<?> getAllUsers(@Param("name") String name) {
        
    //     return ResponseEntity.ok(userRepo.findAll());
    // }

    @GetMapping("/users")
    public List<UserDTO> getMethodUsers() {
        return userRepo.findAll();
    }

    @GetMapping("/users/message")
    public List<UserDTO> getMethodUsersMessage(HttpServletRequest request) {

        List<UserDTO> users = userRepo.findAll();
        for (UserDTO userDTO : users) {

            Principal principal = request.getUserPrincipal();
            String a = principal.getName().toString(); 
            
            String b = userDTO.getEmail();
            String c = "";


            if(a.compareTo(b) == 0){
                c = a + b;
            }
    
            if(a.compareTo(b) < 0){
                c = b + a;
            }
    
            if(a.compareTo(b) > 0){
                c = a + b;
            }
    
         
    
            Optional<Chat> room = chatRepo.findByRoom(c);
            if(room.isPresent()){
                
                List<MessagesDTO> message = messageRepo.findByChatRoomId(room.get().getRoomId());

                userDTO.setLastMessage(message.get(message.size()-1).getMessage() + " at " + message.get(message.size()-1).getDate());
                System.out.println("----------------good of the group " + message.get(message.size()-1).getMessage());
                if(message.get(message.size()-1).getSeen() == null){
                    System.out.println("----------------good Wow ");
                    userDTO.setSeenMesssage(false);
                }
                if(message.get(message.size()-1).getSeen() != null){
                    System.out.println("----------------good No ");
                    userDTO.setSeenMesssage(message.get(message.size()-1).getSeen());
                }
                
                
            }else{
                userDTO.setLastMessage("Say Hey");
                System.out.println("----------------good of the group " );
                
            }
        }
        return users;

        
    }

    @GetMapping("/user/email")
    public String getUserEmail(HttpServletRequest request) {
        Principal principal = request.getUserPrincipal();
        return  principal.getName().toString();
    }

    @GetMapping("/current/user")
    public UserDTO getCurrentUser(HttpServletRequest request) {
        Principal principal = request.getUserPrincipal();
        
        Optional<UserDTO> user =  userRepo.findByEmail(principal.getName().toString());
        System.out.println("--------------- x4444 " + user.get().getId());
        return user.get();
    }

    @PostMapping("/upload/photo/aws")
    public String uploadPhotoToAWS(HttpServletRequest request, @RequestParam("photo") MultipartFile file) throws Exception {
        Principal principal = request.getUserPrincipal();
            Optional<UserDTO> user =  userRepo.findByEmail(principal.getName().toString());


        return awsService.uploadPhoto(principal.getName().toString(), file);
    }

    @PostMapping("/download/photo")
    public ResponseEntity<ByteArrayResource> downloadPhoto(HttpServletRequest request) {

        Principal principal = request.getUserPrincipal();
        Optional<UserDTO> user =  userRepo.findByEmail(principal.getName().toString());

        byte[] data = awsService.downloadFile(user.get().getPhotoId());
        ByteArrayResource resource = new ByteArrayResource(data);

        return ResponseEntity
                .ok()
                .header("Content-type", "application/octet-stream")
                .header("Content-disposition", "attachment; filename=\"" + "\"")
                .body(resource);
    }

    @PostMapping("/download/photo/user")
    public ResponseEntity<ByteArrayResource> downloadPhotoUser(HttpServletRequest request, @RequestBody UserDTO user) {

        
        byte[] data = awsService.downloadFile(user.getPhotoId());
        ByteArrayResource resource = new ByteArrayResource(data);

        return ResponseEntity
                .ok()
                .header("Content-type", "application/octet-stream")
                .header("Content-disposition", "attachment; filename=\"" + "\"")
                .body(resource);
    }

    @PostMapping("/chat/history")
    public List<MessagesDTO> getChatHistory(HttpServletRequest request, @RequestBody ChatHelper chatHelper) {

        String a = chatHelper.getEmail();
        String b = chatHelper.getEmail2();
        String c = "";

        Principal principal = request.getUserPrincipal();
        System.out.println(principal.toString());
        System.out.println("--------------- x5555 " + chatHelper.getEmail() + chatHelper.getEmail2());
        if(a.equals(principal.getName().toString())){

            //      An int value: 0 if the string is equal to the other string.
            //      < 0 if the string is lexicographically less than the other string
            //      > 0 if the string is lexicographically greater than the other string (more characters)

        
            if(a.compareTo(b) == 0){
                c = a + b;
            }

            if(a.compareTo(b) < 0){
                c = b + a;
            }

            if(a.compareTo(b) > 0){
                c = a + b;
            }

            System.out.println("-----------x " + c);
            Optional<Chat> room = chatRepo.findByRoom(c);

            
            

            if(room.isPresent()){
                List<MessagesDTO> messages = messageRepo.findByChatRoomId(room.get().getRoomId());

                for (MessagesDTO messagesDTO : messages) {
                    System.out.println("-------------x C ");
                    messagesDTO.setSeen(true);
                    messageRepo.save(messagesDTO);
                }
                System.out.println("-----------x ahsdfhabsdh");
                return messages;
            
            }
        }

        

        return null;


        
    }
    
    @PostMapping("/chat/seen")
    public List<MessagesDTO> getChatSeen(HttpServletRequest request, @RequestBody ChatHelper chatHelper) {

            System.out.println("-------------x b " + chatHelper.getRoomId());
            List<MessagesDTO> messages = messageRepo.findByChatRoomId(chatHelper.getRoomId());

            for (MessagesDTO messagesDTO : messages) {
                System.out.println("-------------x C ");
                messagesDTO.setSeen(true);
                messageRepo.save(messagesDTO);
            }


        return messages;


        
    }
    


    @GetMapping("/files")
    public ResponseEntity<?> getAllFiles() {
        
        return ResponseEntity.ok(filesRepo.findAll());
    }



    @RequestMapping(value = "/username", method = RequestMethod.GET)
    public String currentUserName(HttpServletRequest request) {
        Principal principal = request.getUserPrincipal();

        return principal.getName().toString();
    }


    @GetMapping("/poly")
    public ResponseEntity<?> getNN() {
       
        // notes https://www.baeldung.com/java-9-http-client 
        
        // first we are creating a request using HttpRequest 
        HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("http://127.0.0.1:5000/nn"))
        .version(HttpClient.Version.HTTP_2)
        .GET()
        .build();

        HttpResponse<String> response = null;

		try {
            
			response = HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString());

		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}


        

        
        return ResponseEntity.ok(response.body());
    }


    @PostMapping("/nn/columns")
    public ResponseEntity<?> nnCols(@RequestParam("file") MultipartFile file, HttpServletRequest request2) throws Exception {



        try {

            // Get the file and save it somewhere
            byte[] bytes = file.getBytes();
            Path path = Paths.get("./linear-regression/src/main/res/" + file.getOriginalFilename());
            Files.write(path, bytes);

        
        } catch (IOException e) {
            e.printStackTrace();
        }

        // so this grabs anything from our principals 
        Principal principal = request2.getUserPrincipal();

        
        
        System.out.println("---------- time " + principal.getName().toString());

        HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("http://127.0.0.1:5000/nn/columns"))
        .POST(BodyPublishers.ofByteArray(file.getBytes()))
        .header("file", "hey")
        .header("file2", "hey222")
        .build();

        
    

        
        HttpResponse<String> response = null;

		try {
            
			response = HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString());

		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        
        return ResponseEntity.ok(response.body());
        
        }


    @PostMapping("/poly/image")
    public ResponseEntity<?> getNNImage(@RequestParam(value ="independent") String independent) {
       
        // notes https://www.baeldung.com/java-9-http-client 
        
        // first we are creating a request using HttpRequest 
        System.out.println(independent);
        HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("http://127.0.0.1:5000/nn/image"))
        .header("independent", independent)
        .version(HttpClient.Version.HTTP_2)
        .GET()
        .build();

        HttpResponse<byte[]> response = null;

		try {
            
			response = HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofByteArray());
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}


        

        
        return ResponseEntity.ok(response.body());
    }

    @RequestMapping(path = "/run/Network", method = RequestMethod.POST)
    public ResponseEntity<?> runNeuralNetwork(@RequestParam("file") MultipartFile file, 
        @RequestParam("target") String target, 
        @RequestParam("neurons") String neurons,
        @RequestParam("epochs") String epochs) throws Exception {

        HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("http://127.0.0.1:5000/nn/run"))
        .POST(BodyPublishers.ofByteArray(file.getBytes()))
        .header("target", target)
        .header("neurons", neurons)
        .header("epochs", epochs)
        .build();

        System.out.println("---------- time44444" + target + neurons + epochs);

        HttpResponse<String> response = null;

        try {
    
            response = HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString());
            System.out.println("---------- time44444");
            System.out.println(response.body());
            //response2 = HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString());
            
            // System.out.println("---------- time5555555");
            // System.out.println(response2.body());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        
    
        return ResponseEntity.ok(response.body());
        
    }



}
