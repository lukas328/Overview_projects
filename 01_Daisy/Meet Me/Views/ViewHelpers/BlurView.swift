//
//  BlurView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 17.01.21.
//

import SwiftUI

struct BlurView: UIViewRepresentable {
    
    // based on a standard UIView
    typealias UIViewType = UIView
    
    // make the style customizable (system is prefered to use due to dark and light mode possibilities)
    var style: UIBlurEffect.Style
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: CGRect.zero)
        view.backgroundColor = .clear
        
        // system is important to make this adaptive for light and dark mode
        let blurEffect = UIBlurEffect(style: style)
        let blurView = UIVisualEffectView(effect: blurEffect)
        blurView.translatesAutoresizingMaskIntoConstraints = false
        
        // at is the layout -> 0 is background
        view.insertSubview(blurView, at: 0)
        
        // make the blurView as big as the screen via UIKit constraints
        NSLayoutConstraint.activate([
            blurView.widthAnchor.constraint(equalTo: view.widthAnchor),
            blurView.heightAnchor.constraint(equalTo: view.heightAnchor)
        ])
        
        
        
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        // not used here due to no animations
    }
           
}


