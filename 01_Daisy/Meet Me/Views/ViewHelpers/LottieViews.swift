//
//  LoadingView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 18.02.21.
//

import SwiftUI

struct LoadingView: View {
    
    @Binding var showLoadingScreen: Bool
    
    var body: some View {
        
        if showLoadingScreen {
            LottieView(filename: "pacman-loading", loopMode: .loop)
                .frame(width: 150, height: 150, alignment: .center)
                .padding()
        }
    }
}

struct CheckView: View {
    
    @Binding var showCheckView: Bool
    
    var text: String?
    
    var body: some View {
        
        if showCheckView {
            VStack {
                
                if text != nil {
                    Text(text!)
                        .foregroundColor(.gray)
                        .frame(width: 150)
                }
                
                LottieView(filename: "check-mark", loopMode: .playOnce)
                    .frame(width: 100, height: 100, alignment: .center)
                    .padding()
                    .onAppear {
                        
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.15) {
                            hapticFeedback(feedBackstyle: .success)
                            showCheckView.toggle()
                        }
                }
            }
            .padding()
            .modifier(FrozenWindowModifier())
        }
        
    }
}

struct LoadingView_Previews: PreviewProvider {
    static var previews: some View {
        CheckView(showCheckView: .constant(true))
    }
}
