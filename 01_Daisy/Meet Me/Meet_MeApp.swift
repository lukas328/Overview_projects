//
//  Meet_MeApp.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 08.01.21.
//
//
import SwiftUI
import Firebase
import FirebaseFirestore
import FirebaseAuth



@main
struct Meet_MeApp: App {
    
    @State var startProcessDone: Bool = false
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    //@ObservedObject private var meet_MeAppVM: Meet_MeAppViewModel = Meet_MeAppViewModel()

    
//    init() {
//        FirebaseApp.configure()
//
//        }
    

    
    var body: some Scene {
        
        WindowGroup {
            
            //ProfileCreationView(profileCreationFinished: .constant(false))
            //meet_MeAppVM.CheckUserAccForAutoLogin()
            
            
//            Copy into if for autologin
//            ||  autoLogin()
//            ||  checkUserAccForAutoLogin()startProcessDone ||  checkUserAccForAutoLogin()
//            if startProcessDone ||  checkUserAccForAutoLogin() {

//            TabBarView()
            
//            MainStartView(startUpDone: $startProcessDone)
            
//            LoginNView()
//            YouProfileNView(showYouProfileView: .constant(true))
                MainControllerView()

//            } else {
//                StartView(startProcessDone: $startProcessDone)
//                    .ignoresSafeArea(.keyboard, edges: .bottom)
//
//            }

            //PickerView()
        }
    }
    
//    func checkUserAccForAutoLogin() -> Bool {
//        if Auth.auth().currentUser != nil{
//            return true
//            
//        } else {
//            return false
//
//        }
//    }
    

    
    
    
}






  



