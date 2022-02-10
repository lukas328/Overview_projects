//
//  LogoutViewModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 05.03.21.
//

import Foundation
import PromiseKit
import FirebaseAuth



class LogoutViewModel: ObservableObject {
    
    
    func authSignOut() -> Promise<Void> {
        return Promise { seal in
            let firebaseAuth = Auth.auth()
            do {
                try firebaseAuth.signOut()
            } catch { let error = error
                seal.reject(error)
            }
            //DispatchQueue.main.async {
            seal.fulfill(())
           // }
        }
        
    }
    
}


